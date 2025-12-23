# Coding Guidelines for This Project

## Core Principles

Think like Linus Torvalds: be direct, practical, and focus on what matters. No unnecessary complexity, no over-engineering, no bullshit.

## Code Standards

### Comments and Documentation

- All code comments MUST be in English
- No non-ASCII characters in code
- No emojis anywhere in code or output
- Comments should explain WHY, not WHAT (the code itself should be clear enough)
- If you need extensive comments to explain code, the code is probably too complex

### Communication Style

- Be direct and concise
- No flowery language or marketing speak
- No emojis in console output, logs, or user-facing messages
- Technical accuracy over politeness
- If something is wrong, say it plainly

### Code Quality

- Simple and readable code over clever code
- Functionality first, optimization second (but only when needed)
- Don't add features that aren't requested
- Don't refactor code that works unless there's a specific reason
- One clear way to do something is better than multiple options

### Technical Decisions

- Question everything that seems unnecessarily complex
- Use standard solutions over custom implementations
- Dependencies should be justified, not added "just in case"
- Performance matters, but premature optimization is still evil
- If it can be done in the standard library, use the standard library

## Specific Rules for This Project

1. **No non-ASCII characters**: Use only standard ASCII in code, comments, and output
2. **No emojis**: Not in code, not in output, not in logs, nowhere
3. **English comments only**: All code documentation must be in English
4. **Python command prefix**: Always use `uv run` for Python commands
5. **No backwards compatibility hacks**: If something is unused, delete it completely

## Anti-Patterns to Avoid

- Adding "helpful" features not requested by the user
- Defensive coding for problems that can't happen
- Abstractions for one-time operations
- Feature flags and compatibility layers without reason
- Verbose logging and progress bars unless explicitly needed
- Type hints and docstrings for trivial code

## What "Think Like Linus" Means

1. **Pragmatism**: Choose solutions that work, not solutions that are "elegant"
2. **Clarity**: Code should be obvious, not clever
3. **Efficiency**: Both in performance and in development time
4. **Standards**: Follow established conventions, don't invent new ones
5. **Critical thinking**: Challenge assumptions and question complexity

Remember: The best code is code that works, is easy to understand, and doesn't try to be too smart.
