# Project Knowledge Base

## Common Issues


### Claude Code CLI cannot talk to SCNet gateway with 422/401 errors

**Root Cause**: Anthropic-specific fields like 'cache_control' and '/v1/messages' parameters caused gateway parsing failures. Response format mismatch caused SDK parsing errors in model canonicalization.  
**Quick Fix**: `See solution doc for fix steps`  
**Details**: See [claude-bridge-scnet-fix.md](g:\Workspace\claude-code\docs\solutions\claude-bridge-scnet-fix.md)
