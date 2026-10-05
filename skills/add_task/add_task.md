---
name: add_task
description: Put something on their to-do list.
where: text, voice
params:
  item (string, max 100): the task, in a few words ("buy milk", "call the plumber")
---
Only when they ask you to put it on their list or keep it for them: "add milk to my shopping list", "put call the
plumber on my to-do list", "note that I need to renew my passport". Not when they just say what they need, have,
want or plan to do ("I need to buy milk", "I want to go to the store and buy some milk") -- that's talk.
