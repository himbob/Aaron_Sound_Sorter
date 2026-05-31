# Legacy root commands

These `.command` files used to live in the project root. They were moved here to keep the root folder readable for AI handoff and human maintenance.

Nothing in this folder should be treated as the current release gate unless `AI_READ_THIS_FIRST.md` says so.

Current required smoke gate for behavior-changing sorter work:

```bash
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE.command
```
