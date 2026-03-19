You are running one iteration of an automated bug sweep for this repository.

Your job on each iteration is to:

1. Inspect open bug issues in the target GitHub repository.
2. Inspect prior bug-sweep reports from `docs/test-reports/agentic-bug-sweep/`.
3. Choose the next unexplored or highest-yield area to investigate.
4. Use the installed `test-feature` skill from `agentic-tests` to investigate that area.
5. Decide whether the result means:
   - `create`: a new issue should be created
   - `update`: an existing issue should be updated
   - `merge`: the finding is the same bug as an existing canonical issue and duplicates should be closed
   - `none`: no actionable bug was found

Relationship rules:

- If the finding is the same bug as an existing issue, use `merge`.
- If the finding is not the same bug, but it likely shares the same root cause as an existing issue, keep the primary action as `create` or `update` and populate `related_issue_numbers`.
- Only use `duplicates_to_close` for true duplicates of the same bug.

Output rules:

- Return only JSON that matches the provided schema.
- Always include a short `summary` and the generated `report_path`.
- The schema requires every top-level field to be present. Use `null` for fields that are irrelevant to the chosen action.
- For new issues, provide `issue.title`, `issue.body`, and `issue.labels`.
- For issue updates, provide `canonical_issue_number` and `issue_comment`.
- For duplicate consolidation, provide `canonical_issue_number`, `issue_comment`, `duplicates_to_close`, and `duplicate_comment`.
- If you provide `related_issue_numbers`, also provide `related_comment`.

Do not run raw GitHub issue mutations yourself. The shell script will apply any create, update, merge, or related-issue actions.
