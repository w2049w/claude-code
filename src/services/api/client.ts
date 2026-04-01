import Anthropic, { type ClientOptions } from '@anthropic-ai/sdk'
import { randomUUID } from 'crypto'
import type { GoogleAuth } from 'google-auth-library'
import {
  checkAndRefreshOAuthTokenIfNeeded,
  getAnthropicApiKey,
  getApiKeyFromApiKeyHelper,
  getClaudeAIOAuthTokens,
  isClaudeAISubscriber,
  refreshAndGetAwsCredentials,
  refreshGcpCredentialsIfNeeded,
} from 'src/utils/auth.js'
import { getUserAgent } from 'src/utils/http.js'
import { getSmallFastModel } from 'src/utils/model/model.js'
import {
  getAPIProvider,
  isFirstPartyAnthropicBaseUrl,
} from 'src/utils/model/providers.js'
import { getProxyFetchOptions } from 'src/utils/proxy.js'
import {
  getIsNonInteractiveSession,
  getSessionId,
} from '../../bootstrap/state.js'
import { getOauthConfig } from '../../constants/oauth.js'
import { isDebugToStdErr, logForDebugging } from '../../utils/debug.js'
import {
  getAWSRegion,
  getVertexRegionForModel,
  isEnvTruthy,
} from '../../utils/envUtils.js'

/**
 * Environment variables for different client types:
 *
 * Direct API:
 * - ANTHROPIC_API_KEY: Required for direct API access
 *
 * AWS Bedrock:
 * - AWS credentials configured via aws-sdk defaults
 * - AWS_REGION or AWS_DEFAULT_REGION: Sets the AWS region for all models (default: us-east-1)
 * - ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION: Optional. Override AWS region specifically for the small fast model (Haiku)
 *
 * Foundry (Azure):
 * - ANTHROPIC_FOUNDRY_RESOURCE: Your Azure resource name (e.g., 'my-resource')
 *   For the full endpoint: https://{resource}.services.ai.azure.com/anthropic/v1/messages
 * - ANTHROPIC_FOUNDRY_BASE_URL: Optional. Alternative to resource - provide full base URL directly
 *   (e.g., 'https://my-resource.services.ai.azure.com')
 *
 * Authentication (one of the following):
 * - ANTHROPIC_FOUNDRY_API_KEY: Your Microsoft Foundry API key (if using API key auth)
 * - Azure AD authentication: If no API key is provided, uses DefaultAzureCredential
 *   which supports multiple auth methods (environment variables, managed identity,
 *   Azure CLI, etc.). See: https://docs.microsoft.com/en-us/javascript/api/@azure/identity
 *
 * Vertex AI:
 * - Model-specific region variables (highest priority):
 *   - VERTEX_REGION_CLAUDE_3_5_HAIKU: Region for Claude 3.5 Haiku model
 *   - VERTEX_REGION_CLAUDE_HAIKU_4_5: Region for Claude Haiku 4.5 model
 *   - VERTEX_REGION_CLAUDE_3_5_SONNET: Region for Claude 3.5 Sonnet model
 *   - VERTEX_REGION_CLAUDE_3_7_SONNET: Region for Claude 3.7 Sonnet model
 * - CLOUD_ML_REGION: Optional. The default GCP region to use for all models
 *   If specific model region not specified above
 * - ANTHROPIC_VERTEX_PROJECT_ID: Required. Your GCP project ID
 * - Standard GCP credentials configured via google-auth-library
 *
 * Priority for determining region:
 * 1. Hardcoded model-specific environment variables
 * 2. Global CLOUD_ML_REGION variable
 * 3. Default region from config
 * 4. Fallback region (us-east5)
 */

function createStderrLogger(): ClientOptions['logger'] {
  return {
    error: (msg, ...args) =>
      // biome-ignore lint/suspicious/noConsole:: intentional console output -- SDK logger must use console
      console.error('[Anthropic SDK ERROR]', msg, ...args),
    // biome-ignore lint/suspicious/noConsole:: intentional console output -- SDK logger must use console
    warn: (msg, ...args) => console.error('[Anthropic SDK WARN]', msg, ...args),
    // biome-ignore lint/suspicious/noConsole:: intentional console output -- SDK logger must use console
    info: (msg, ...args) => console.error('[Anthropic SDK INFO]', msg, ...args),
    debug: (msg, ...args) =>
      // biome-ignore lint/suspicious/noConsole:: intentional console output -- SDK logger must use console
      console.error('[Anthropic SDK DEBUG]', msg, ...args),
  }
}

export async function getAnthropicClient({
  apiKey,
  maxRetries,
  model,
  fetchOverride,
  source,
}: {
  apiKey?: string
  maxRetries: number
  model?: string
  fetchOverride?: ClientOptions['fetch']
  source?: string
}): Promise<Anthropic> {
  const logMsg = `[DEBUG-FATAL] getAnthropicClient CALLED from ${source || 'unknown'} at ${new Date().toISOString()}\n`;
  try {
    const fs = require('node:fs');
    fs.appendFileSync('bridge.log', logMsg);
  } catch (e) {
    // Fallback if require fails in some environments
    console.log(`Log failed: ${e}`);
  }
  console.log(`\x1b[42m\x1b[30m${logMsg.trim()}\x1b[0m`);
  const containerId = process.env.CLAUDE_CODE_CONTAINER_ID
  const remoteSessionId = process.env.CLAUDE_CODE_REMOTE_SESSION_ID
  const clientApp = process.env.CLAUDE_AGENT_SDK_CLIENT_APP
  const customHeaders = getCustomHeaders()
  const defaultHeaders: { [key: string]: string } = {
    'x-app': 'cli',
    'User-Agent': getUserAgent(),
    'X-Claude-Code-Session-Id': getSessionId(),
    ...customHeaders,
    ...(containerId ? { 'x-claude-remote-container-id': containerId } : {}),
    ...(remoteSessionId
      ? { 'x-claude-remote-session-id': remoteSessionId }
      : {}),
    // SDK consumers can identify their app/library for backend analytics
    ...(clientApp ? { 'x-client-app': clientApp } : {}),
  }

  // Log API client configuration for HFI debugging
  logForDebugging(
    `[API:request] Creating client, ANTHROPIC_CUSTOM_HEADERS present: ${!!process.env.ANTHROPIC_CUSTOM_HEADERS}, has Authorization header: ${!!customHeaders['Authorization']}`,
  )

  // Add additional protection header if enabled via env var
  const additionalProtectionEnabled = isEnvTruthy(
    process.env.CLAUDE_CODE_ADDITIONAL_PROTECTION,
  )
  if (additionalProtectionEnabled) {
    defaultHeaders['x-anthropic-additional-protection'] = 'true'
  }

  logForDebugging('[API:auth] OAuth token check starting')
  await checkAndRefreshOAuthTokenIfNeeded()
  logForDebugging('[API:auth] OAuth token check complete')

  const resolvedApiKey = isClaudeAISubscriber() ? undefined : apiKey || getAnthropicApiKey()
  const resolvedFetch = buildFetch(fetchOverride, source, resolvedApiKey)

  const ARGS = {
    defaultHeaders,
    maxRetries,

    timeout: parseInt(process.env.API_TIMEOUT_MS || String(600 * 1000), 10),
    dangerouslyAllowBrowser: true,
    fetchOptions: getProxyFetchOptions({
      forAnthropicAPI: true,
    }) as ClientOptions['fetchOptions'],
    ...(resolvedFetch && {
      fetch: resolvedFetch,
    }),
  }
  if (isEnvTruthy(process.env.CLAUDE_CODE_USE_BEDROCK)) {
    const { AnthropicBedrock } = await import('@anthropic-ai/bedrock-sdk')
    // Use region override for small fast model if specified
    const awsRegion =
      model === getSmallFastModel() &&
        process.env.ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION
        ? process.env.ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION
        : getAWSRegion()

    const bedrockArgs: Record<string, unknown> = {
      ...ARGS,
      awsRegion,
      ...(isEnvTruthy(process.env.CLAUDE_CODE_SKIP_BEDROCK_AUTH) && {
        skipAuth: true,
      }),
      ...(isDebugToStdErr() && { logger: createStderrLogger() }),
    }

    // Add API key authentication if available
    if (process.env.AWS_BEARER_TOKEN_BEDROCK) {
      bedrockArgs.skipAuth = true
      // Add the Bearer token for Bedrock API key authentication
      bedrockArgs.defaultHeaders = {
        ...(bedrockArgs.defaultHeaders as Record<string, string> | undefined),
        Authorization: `Bearer ${process.env.AWS_BEARER_TOKEN_BEDROCK}`,
      }
    } else if (!isEnvTruthy(process.env.CLAUDE_CODE_SKIP_BEDROCK_AUTH)) {
      // Refresh auth and get credentials with cache clearing
      const cachedCredentials = await refreshAndGetAwsCredentials()
      if (cachedCredentials) {
        bedrockArgs.awsAccessKey = cachedCredentials.accessKeyId
        bedrockArgs.awsSecretKey = cachedCredentials.secretAccessKey
        bedrockArgs.awsSessionToken = cachedCredentials.sessionToken
      }
    }
    // we have always been lying about the return type - this doesn't support batching or models
    return new AnthropicBedrock(bedrockArgs) as unknown as Anthropic
  }
  if (isEnvTruthy(process.env.CLAUDE_CODE_USE_FOUNDRY)) {
    const { AnthropicFoundry } = await import('@anthropic-ai/foundry-sdk')
    // Determine Azure AD token provider based on configuration
    // SDK reads ANTHROPIC_FOUNDRY_API_KEY by default
    let azureADTokenProvider: (() => Promise<string>) | undefined
    if (!process.env.ANTHROPIC_FOUNDRY_API_KEY) {
      if (isEnvTruthy(process.env.CLAUDE_CODE_SKIP_FOUNDRY_AUTH)) {
        // Mock token provider for testing/proxy scenarios (similar to Vertex mock GoogleAuth)
        azureADTokenProvider = () => Promise.resolve('')
      } else {
        // Use real Azure AD authentication with DefaultAzureCredential
        const {
          DefaultAzureCredential: AzureCredential,
          getBearerTokenProvider,
        } = await import('@azure/identity')
        azureADTokenProvider = getBearerTokenProvider(
          new AzureCredential(),
          'https://cognitiveservices.azure.com/.default',
        )
      }
    }

    const foundryArgs: ConstructorParameters<typeof AnthropicFoundry>[0] = {
      ...ARGS,
      ...(azureADTokenProvider && { azureADTokenProvider }),
      ...(isDebugToStdErr() && { logger: createStderrLogger() }),
    }
    // we have always been lying about the return type - this doesn't support batching or models
    return new AnthropicFoundry(foundryArgs) as unknown as Anthropic
  }
  if (isEnvTruthy(process.env.CLAUDE_CODE_USE_VERTEX)) {
    // Refresh GCP credentials if gcpAuthRefresh is configured and credentials are expired
    // This is similar to how we handle AWS credential refresh for Bedrock
    if (!isEnvTruthy(process.env.CLAUDE_CODE_SKIP_VERTEX_AUTH)) {
      await refreshGcpCredentialsIfNeeded()
    }

    const [{ AnthropicVertex }, { GoogleAuth }] = await Promise.all([
      import('@anthropic-ai/vertex-sdk'),
      import('google-auth-library'),
    ])
    // TODO: Cache either GoogleAuth instance or AuthClient to improve performance
    // Currently we create a new GoogleAuth instance for every getAnthropicClient() call
    // This could cause repeated authentication flows and metadata server checks
    // However, caching needs careful handling of:
    // - Credential refresh/expiration
    // - Environment variable changes (GOOGLE_APPLICATION_CREDENTIALS, project vars)
    // - Cross-request auth state management
    // See: https://github.com/googleapis/google-auth-library-nodejs/issues/390 for caching challenges

    // Prevent metadata server timeout by providing projectId as fallback
    // google-auth-library checks project ID in this order:
    // 1. Environment variables (GCLOUD_PROJECT, GOOGLE_CLOUD_PROJECT, etc.)
    // 2. Credential files (service account JSON, ADC file)
    // 3. gcloud config
    // 4. GCE metadata server (causes 12s timeout outside GCP)
    //
    // We only set projectId if user hasn't configured other discovery methods
    // to avoid interfering with their existing auth setup

    // Check project environment variables in same order as google-auth-library
    // See: https://github.com/googleapis/google-auth-library-nodejs/blob/main/src/auth/googleauth.ts
    const hasProjectEnvVar =
      process.env['GCLOUD_PROJECT'] ||
      process.env['GOOGLE_CLOUD_PROJECT'] ||
      process.env['gcloud_project'] ||
      process.env['google_cloud_project']

    // Check for credential file paths (service account or ADC)
    // Note: We're checking both standard and lowercase variants to be safe,
    // though we should verify what google-auth-library actually checks
    const hasKeyFile =
      process.env['GOOGLE_APPLICATION_CREDENTIALS'] ||
      process.env['google_application_credentials']

    const googleAuth = isEnvTruthy(process.env.CLAUDE_CODE_SKIP_VERTEX_AUTH)
      ? ({
        // Mock GoogleAuth for testing/proxy scenarios
        getClient: () => ({
          getRequestHeaders: () => ({}),
        }),
      } as unknown as GoogleAuth)
      : new GoogleAuth({
        scopes: ['https://www.googleapis.com/auth/cloud-platform'],
        // Only use ANTHROPIC_VERTEX_PROJECT_ID as last resort fallback
        // This prevents the 12-second metadata server timeout when:
        // - No project env vars are set AND
        // - No credential keyfile is specified AND
        // - ADC file exists but lacks project_id field
        //
        // Risk: If auth project != API target project, this could cause billing/audit issues
        // Mitigation: Users can set GOOGLE_CLOUD_PROJECT to override
        ...(hasProjectEnvVar || hasKeyFile
          ? {}
          : {
            projectId: process.env.ANTHROPIC_VERTEX_PROJECT_ID,
          }),
      })

    const vertexArgs: ConstructorParameters<typeof AnthropicVertex>[0] = {
      ...ARGS,
      region: getVertexRegionForModel(model),
      googleAuth: googleAuth as any,
      ...(isDebugToStdErr() && { logger: createStderrLogger() }),
    }
    // we have always been lying about the return type - this doesn't support batching or models
    return new AnthropicVertex(vertexArgs) as unknown as Anthropic
  }

  // Determine authentication method based on available tokens
  const provider = getAPIProvider()
  const clientConfig: ConstructorParameters<typeof Anthropic>[0] = {
    apiKey: resolvedApiKey ?? null,
    authToken: isClaudeAISubscriber()
      ? getClaudeAIOAuthTokens()?.accessToken
      : undefined,
    // Set baseURL from OAuth config when using staging OAuth
    ...(process.env.USER_TYPE === 'ant' &&
      isEnvTruthy(process.env.USE_STAGING_OAUTH)
      ? { baseURL: getOauthConfig().BASE_API_URL }
      : {}),
    ...ARGS,
    ...(isDebugToStdErr() && { logger: createStderrLogger() }),
    ...(provider === 'openai-compatible' && {
      // Force base URL if detected as openai-compatible to ensure it has correct protocol
      baseURL: process.env.ANTHROPIC_BASE_URL
    })
  }



  return new Anthropic(clientConfig)
}

async function configureApiKeyHeaders(
  headers: Record<string, string>,
  isNonInteractiveSession: boolean,
): Promise<void> {
  const token =
    process.env.ANTHROPIC_AUTH_TOKEN ||
    (await getApiKeyFromApiKeyHelper(isNonInteractiveSession))
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
}

function getCustomHeaders(): Record<string, string> {
  const customHeaders: Record<string, string> = {}
  const customHeadersEnv = process.env.ANTHROPIC_CUSTOM_HEADERS

  if (!customHeadersEnv) return customHeaders

  // Split by newlines to support multiple headers
  const headerStrings = customHeadersEnv.split(/\n|\r\n/)

  for (const headerString of headerStrings) {
    if (!headerString.trim()) continue

    // Parse header in format "Name: Value" (curl style). Split on first `:`
    // then trim — avoids regex backtracking on malformed long header lines.
    const colonIdx = headerString.indexOf(':')
    if (colonIdx === -1) continue
    const name = headerString.slice(0, colonIdx).trim()
    const value = headerString.slice(colonIdx + 1).trim()
    if (name) {
      customHeaders[name] = value
    }
  }

  return customHeaders
}

export const CLIENT_REQUEST_ID_HEADER = 'x-client-request-id'

function buildFetch(
  fetchOverride: ClientOptions['fetch'],
  _source: string | undefined,
  apiKey?: string
): ClientOptions['fetch'] {
  const inner = fetchOverride ?? (globalThis.fetch as any);
  // Only send to the first-party API — Bedrock/Vertex/Foundry don't log it
  // and unknown headers risk rejection by strict proxies (inc-4029 class).
  const injectClientRequestId =
    getAPIProvider() === 'firstParty' && isFirstPartyAnthropicBaseUrl()
  return async (input, init) => {

    // eslint-disable-next-line eslint-plugin-n/no-unsupported-features/node-builtins
    const headers = new Headers(init?.headers)
    // Generate a client-side request ID so timeouts (which return no server
    // request ID) can still be correlated with server logs by the API team.
    // Callers that want to track the ID themselves can pre-set the header.
    if (injectClientRequestId && !headers.has(CLIENT_REQUEST_ID_HEADER)) {
      headers.set(CLIENT_REQUEST_ID_HEADER, randomUUID())
    }
    let currentProvider = getAPIProvider()
    const envBaseUrl = process.env.ANTHROPIC_BASE_URL
    const url = input instanceof Request ? input.url : String(input);

    // ROBUSTNESS: If URL is non-official, force openai-compatible provider
    const isOfficialUrl = url.includes('anthropic.com') ||
      url.includes('googleapis.com') ||
      url.includes('amazonaws.com');

    if (currentProvider === 'firstParty' && !isOfficialUrl && envBaseUrl) {
      currentProvider = 'openai-compatible'
      console.log(`\x1b[45m\x1b[37m[DEBUG-FATAL] FORCING openai-compatible due to non-official URL: ${url}\x1b[0m`);
    }

    // HEARTBEAT & DIAGNOSTICS
    console.log(`\x1b[43m\x1b[30m[DEBUG-FATAL] Provider: ${currentProvider}\x1b[0m`);
    console.log(`\x1b[43m\x1b[30m[DEBUG-FATAL] BASE_URL: ${envBaseUrl}\x1b[0m`);
    console.log(`\x1b[43m\x1b[30m[DEBUG-FATAL] TARGET_URL: ${url}\x1b[0m`);

    if (currentProvider === 'openai-compatible') {
      const keyToUse = apiKey || process.env.ANTHROPIC_API_KEY
      console.log(`\x1b[43m\x1b[30m[DEBUG-FATAL] Key length: ${keyToUse?.length || 0}\x1b[0m`);

      // The Bridge: Intercept and Map Protocols
      const bridgeFetch = async (input: RequestInfo | URL, init?: RequestInit) => {
        let url = input instanceof Request ? input.url : String(input)
        const logIntercept = `[BRIDGE] INTERCEPTED: ${url} at ${new Date().toISOString()}\n`;
        try {
          const fs = require('node:fs');
          fs.appendFileSync('bridge.log', logIntercept);
        } catch (e) {
          // Fallback
        }
        console.log(`\x1b[44m\x1b[37m${logIntercept.trim()}\x1b[0m`);

        // eslint-disable-next-line eslint-plugin-n/no-unsupported-features/node-builtins
        const newHeaders = new Headers(init?.headers)

        if (keyToUse) {
          newHeaders.set('Authorization', `Bearer ${keyToUse}`)
          console.log(`\x1b[44m\x1b[37m[BRIDGE] Set Auth Header (Len: ${keyToUse.length})\x1b[0m`)
        }

        // Log headers for debugging (keys only for security)
        const headerKeys = Array.from(newHeaders.keys()).join(', ');
        const logHeaders = `[BRIDGE] HEADERS SENT: ${headerKeys} at ${new Date().toISOString()}\n`;
        try { require('node:fs').appendFileSync('bridge.log', logHeaders); } catch (e) {}

        // Force cleanup of Anthropic-only headers that freak out simple proxies
        newHeaders.delete('x-api-key')
        newHeaders.delete('anthropic-beta')
        newHeaders.delete('anthropic-version')
        newHeaders.delete('x-app')
        newHeaders.delete('x-claude-code-session-id')

        // REWRITE ENDPOINT: Handle both path mapping and potential /v1/v1 duplicates
        const baseUrlObj = new URL(url)
        let path = baseUrlObj.pathname
        console.log(`\x1b[44m\x1b[37m[BRIDGE] ORIG PATH: ${path}\x1b[0m`)
        
        if (path.includes('/v1/messages')) {
            path = path.replace(/\/v1\/(v1\/)?messages$/, '/v1/chat/completions')
            url = `${baseUrlObj.origin}${path}` // This naturally strips the query string
            const logRewritten = `[BRIDGE] REWRITTEN TO: ${url} at ${new Date().toISOString()}\n`;
            try { require('node:fs').appendFileSync('bridge.log', logRewritten); } catch (e) {}
            console.log(`\x1b[44m\x1b[37m${logRewritten.trim()}\x1b[0m`)
        } else if (path.includes('/v1/v1/')) {
            path = path.replace(/\/v1\/v1\//, '/v1/')
            url = `${baseUrlObj.origin}${path}`
            const logNormalized = `[BRIDGE] NORMALIZED TO: ${url} at ${new Date().toISOString()}\n`;
            try { require('node:fs').appendFileSync('bridge.log', logNormalized); } catch (e) {}
            console.log(`\x1b[44m\x1b[37m${logNormalized.trim()}\x1b[0m`)
        }

        const response = await (async () => {
          if (init?.body && (init.method === 'POST' || !init.method)) {
            try {
              const bodyText = typeof init.body === 'string' ? init.body : new TextDecoder().decode(init.body as BufferSource)
              const bodyJson = JSON.parse(bodyText)
              
              // 1. TRANSFORM BODY: Convert Anthropic 'system' property
              if (bodyJson.system) {
                  const systemContent = Array.isArray(bodyJson.system) 
                      ? bodyJson.system.map((p: any) => typeof p === 'string' ? p : p.text).join('\n')
                      : bodyJson.system;
                  
                  if (!Array.isArray(bodyJson.messages)) bodyJson.messages = [];
                  bodyJson.messages.unshift({ role: 'system', content: systemContent })
                  delete bodyJson.system
              }

              // 2. SANITIZE MESSAGES: Flatten content and remove Anthropic-specific extensions
              if (Array.isArray(bodyJson.messages)) {
                  bodyJson.messages = bodyJson.messages.map((msg: any) => {
                      let textContent = '';
                      if (Array.isArray(msg.content)) {
                          textContent = msg.content.map((p: any) => typeof p === 'string' ? p : (p.text || '')).join('\n');
                      } else {
                          textContent = String(msg.content || '');
                      }
                      return { role: msg.role, content: textContent };
                  });
              }

              // 3. WHUTELIST OpenAI fields & RENAME others
              const openAiBody: any = {
                  model: bodyJson.model,
                  messages: bodyJson.messages,
                  temperature: bodyJson.temperature ?? 0.7,
                  top_p: bodyJson.top_p ?? 1.0,
                  stream: false
              };

              // OpenAI prefers max_completion_tokens or max_tokens
              // We'll provide both to be safe, or just one if the gateway is picky
              openAiBody.max_tokens = 4096; 

              if (bodyJson.stop_sequences) {
                  openAiBody.stop = bodyJson.stop_sequences;
              }

              // OVERRIDE MODEL (Robustness)
              const envModel = process.env.ANTHROPIC_MODEL;
              let targetModel = envModel ? envModel.replace(/^["']|["']$/g, '') : undefined;
              try {
                  const envText = require('node:fs').readFileSync('.env', 'utf8');
                  const match = envText.match(/^ANTHROPIC_MODEL=(.+)$/m);
                  if (match) targetModel = match[1].trim().replace(/^["']|["']$/g, '');
              } catch (e) {}

              if (targetModel) {
                  openAiBody.model = targetModel;
              } else if (typeof openAiBody.model === 'string') {
                  openAiBody.model = openAiBody.model.replace(/^["']|["']$/g, '');
              }

              const finalBody = JSON.stringify(openAiBody)
              // Log FULL body for one final check
              const logFinal = `[BRIDGE] SENDING CLEAN BODY: ${JSON.stringify(openAiBody, null, 2)} at ${new Date().toISOString()}\n`;
              try { require('node:fs').appendFileSync('bridge.log', logFinal); } catch (e) {}
              
              // Only log first 200 chars to console to avoid cluttering TUI
              console.log(`\x1b[44m\x1b[37m[BRIDGE] Sending ${openAiBody.model} request...\x1b[0m`)

              const res = await inner(url, {
                ...init,
                headers: newHeaders,
                body: finalBody
              })

              const logStatus = `[BRIDGE] RESPONSE STATUS: ${res.status} at ${new Date().toISOString()}\n`;
              try { require('node:fs').appendFileSync('bridge.log', logStatus); } catch (e) {}

              if (!res.ok) {
                  const errorText = await res.clone().text();
                  const logErrorBody = `[BRIDGE] ERROR RESPONSE BODY: ${errorText} at ${new Date().toISOString()}\n`;
                  try { require('node:fs').appendFileSync('bridge.log', logErrorBody); } catch (e) {}
                  console.log(`\x1b[41m\x1b[37m${logErrorBody.trim()}\x1b[0m`)
                  return res;
              }

              // BRIDGE RESPONSE: Convert OpenAI response back to Anthropic format
              try {
                  const clonedRes = res.clone();
                  const openAiRes = await clonedRes.json();
                  
                  if (openAiRes.choices && openAiRes.choices[0]) {
                      const choice = openAiRes.choices[0];
                      const anthropicRes = {
                          id: openAiRes.id || `msg_bridge_${Date.now()}`,
                          type: "message",
                          role: "assistant",
                          model: openAiRes.model || targetModel || "MiniMax-M2.5",
                          content: [
                              {
                                  type: "text",
                                  text: choice.message?.content || ""
                              }
                          ],
                          stop_reason: choice.finish_reason === "stop" ? "end_turn" : choice.finish_reason,
                          stop_sequence: null,
                          usage: {
                              input_tokens: openAiRes.usage?.prompt_tokens || 0,
                              output_tokens: openAiRes.usage?.completion_tokens || 0
                          }
                      };

                      // If there are tool calls, we need to map them too (CRITICAL for Claude Code)
                      if (choice.message?.tool_calls) {
                          for (const tc of choice.message.tool_calls) {
                              if (tc.type === 'function') {
                                  anthropicRes.content.push({
                                      type: "tool_use",
                                      id: tc.id,
                                      name: tc.function.name,
                                      input: JSON.parse(tc.function.arguments)
                                  } as any);
                              }
                          }
                      }

                      const logBridgeRes = `[BRIDGE] CONVERTED RESPONSE: ${JSON.stringify(anthropicRes, null, 2)} at ${new Date().toISOString()}\n`;
                      try { require('node:fs').appendFileSync('bridge.log', logBridgeRes); } catch (e) {}

                      return new Response(JSON.stringify(anthropicRes), {
                          status: res.status,
                          statusText: res.statusText,
                          headers: res.headers
                      });
                  }
              } catch (bridgeErr) {
                  const logBridgeErr = `[BRIDGE] RESPONSE BRIDGE FAILED: ${bridgeErr} at ${new Date().toISOString()}\n`;
                  try { require('node:fs').appendFileSync('bridge.log', logBridgeErr); } catch (e) {}
              }

              return res;

            } catch (e) {
              console.log(`\x1b[41m\x1b[37m[BRIDGE] Body transform failed: ${e}\x1b[0m`)
            }
          }
          return await inner(url, { ...init, headers: newHeaders })
        })()

        const logStatus = `[BRIDGE] RESPONSE STATUS: ${response.status} at ${new Date().toISOString()}\n`;
        try { require('node:fs').appendFileSync('bridge.log', logStatus); } catch (e) {}
        console.log(`\x1b[44m\x1b[37m${logStatus.trim()}\x1b[0m`)

        // Handle Streaming: Translate OpenAI SSE -> Anthropic SSE
        if (response.ok && response.headers.get('content-type')?.includes('text/event-stream')) {
          // ... (keep the existing stream logic)
          return new Response(response.body?.pipeThrough(new TransformStream({
            start(controller) {
              const encoder = new TextEncoder()
              controller.enqueue(encoder.encode(`event: message_start\ndata: ${JSON.stringify({
                type: 'message_start',
                message: { id: `msg_shim_${Date.now()}`, type: 'message', role: 'assistant', content: [], model: 'minimax', stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } }
              })}\n\n`))
              controller.enqueue(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify({
                type: 'content_block_start',
                index: 0,
                content_block: { type: 'text', text: '' }
              })}\n\n`))
            },
            transform(chunk, controller) {
              const decoder = new TextDecoder()
              const encoder = new TextEncoder()
              const text = decoder.decode(chunk)
              for (const line of text.split('\n')) {
                if (line.startsWith('data: ')) {
                  const dataStr = line.slice(6).trim()
                  if (dataStr === '[DONE]') continue
                  try {
                    const openAiData = JSON.parse(dataStr)
                    const content = openAiData.choices?.[0]?.delta?.content || ''
                    if (content) {
                      controller.enqueue(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify({
                        type: 'content_block_delta',
                        index: 0,
                        delta: { type: 'text_delta', text: content }
                      })}\n\n`))
                    }
                  } catch { }
                }
              }
            },
            flush(controller) {
              const encoder = new TextEncoder()
              controller.enqueue(encoder.encode(`event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`))
            }
          })), {
            headers: response.headers,
            status: response.status,
            statusText: response.statusText
          })
        }

        return response
      }

      return bridgeFetch(input, { ...init, headers })
    }

    try {
      const url = input instanceof Request ? input.url : String(input)
      const id = headers.get(CLIENT_REQUEST_ID_HEADER)
      logForDebugging(`[API:OPENAI_PATCH] Routed: ${url}`)
    } catch { }

    return inner(input, { ...init, headers })
  }
}


