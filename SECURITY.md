# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's security advisory feature rather than opening a public issue containing exploit details or credentials.

## Credential handling

- Store API keys in environment variables or a local `.env` file.
- Never commit `.env`, tokens, local vector stores, experiment logs, or database files.
- The repository includes `.env.example` with empty placeholders only.
- If a credential is committed accidentally, revoke it immediately and remove it from Git history before publishing.

## Data handling

The benchmark operates on third-party datasets and may generate large transformed traces. Verify dataset terms before redistributing derived records. This repository distributes code only: do not commit source records, derived samples, model answers, aggregate tables, or generated figures.

