---
name: remind
description: Remind them of something later -- at a time, or in a while.
where: text, voice
params:
  at (string): when -- a clock time as HH:MM on a 24-hour clock, or "in N minutes" / "in N hours"
  about (string, max 200): what to remind them of
---
Use when they ask you to remind them of something later: "remind me at 5 to call mom", "ping me at 18:30 about
dinner", "remind me in 20 minutes to check the oven". Asked by text, you'll text it to them; out loud, you'll say it
at home. Not for "remind me what you said" or "remind me your name" -- those are questions.
