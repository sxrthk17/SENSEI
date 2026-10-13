
from memory_db import (
    init_db,
    init_learning_db,
    save_pomodoro_session,
    get_pomodoro_stats,
    add_todo_task,
    get_todo_tasks,
    set_todo_completed,
)

init_db()
init_learning_db()

# Test a completed focus session
save_pomodoro_session(
    planned_minutes=25,
    completed_minutes=25,
    session_type="focus",
)

print("Pomodoro stats:", get_pomodoro_stats())

# Test task creation and completion
task_id = add_todo_task(
    "Revise DAA",
    category="College",
    priority="High",
)

print("Tasks:", get_todo_tasks())

set_todo_completed(task_id, True)
print("After completion:", get_todo_tasks())
