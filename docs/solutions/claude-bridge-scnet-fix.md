---
symptoms: ["Claude Code CLI cannot talk to SCNet gateway with 422/401 errors", "JavaScript crash 'undefined.toLowerCase' after getting response"]
root_cause: "Anthropic-specific fields like 'cache_control' and '/v1/messages' parameters caused gateway parsing failures. Response format mismatch caused SDK parsing errors in model canonicalization."
module: "api-client"
severity: "high"
tags: ["bridge", "openai", "protocol-mapping", "crash-fix", "headless"]
---

# Claude Code to SCNet/OpenAI Bridge Solution

This document details the fix for bridging Claude Code (Anthropic SDK) to an OpenAI-compatible gateway (SCNet/MiniMax) and resolving a critical JavaScript runtime crash during response processing.

## Context

Claude Code is designed for Anthropic's native API. When redirected to `https://api.scnet.cn/api/llm`, the gateway rejected requests due to:
1. Unknown query parameters (`?beta=true`).
2. Anthropic-specific request body fields (`cache_control`, `system` as a top-level field).
3. Authorization header mismatch (Anthropic uses `x-api-key`, OpenAI uses `Bearer`).

Additionally, receiving a `choices`-based OpenAI response caused the client-side `model.ts` to crash when attempting to canonicalize unrecognized model names.

## Working Solution

The solution involved a dual-layer fix: a request/response interceptor in the API client and a defensive patch in the model utility.

### 1. Unified Protocol Bridge & Auth Override

Modified `src/services/api/client.ts` by implementing `bridgeFetch` to achieve protocol mapping and bypass the native login requirements:

- **Authentication Hijack**: Forced injection of `Authorization: Bearer <key>` using the `ANTHROPIC_API_KEY` environment variable.
- **Header Stripping**: Explicitly removed all Anthropic-branded headers (like `x-api-key`) that were causing the SCNet gateway to reject sessions as "not logged in" (401 error).
- **Request Formatting**:

```typescript
// Request Transformation
const openAiBody = {
    model: targetModel,
    messages: sanitizedMessages,
    max_tokens: 4096,
    stream: false,
    // ... other OpenAI fields
};

// Response Transformation
const anthropicRes = {
    id: openAiRes.id,
    type: "message",
    role: "assistant",
    model: openAiRes.model,
    content: [{ type: "text", text: choice.message.content }],
    usage: {
        input_tokens: openAiRes.usage.prompt_tokens,
        output_tokens: openAiRes.usage.completion_tokens
    }
};
```

### 2. Defensive Model Canonicalization

Patched `src/utils/model/model.ts` to prevent the `undefined` error:

```typescript
export function firstPartyNameToCanonical(name: ModelName): ModelShortName {
  if (!name) {
    return 'claude-3-5-sonnet' as any; // Fallback
  }
  name = name.toLowerCase() 
  // ...
}
```

### 3. Implementation Steps

1. **Verify Environment**: Set `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_MODEL` in `.env`.
2. **Apply Bridge**: Replace the standard `fetch` call in `client.ts` with the intercepted `bridgeFetch`.
3. **Apply Crash Fix**: Add null checks to `firstPartyNameToCanonical` in `model.ts`.

## Verification Results

- ✅ `200 OK` from SCNet gateway.
- ✅ Correct token usage tracking.
- ✅ No JavaScript crashes after message display.
- ✅ Supports both basic chat and tool usage mapping.
