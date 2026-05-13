import neostandard, { resolveIgnoresFromGitignore } from 'neostandard'

const ignores = resolveIgnoresFromGitignore()
export default [
  ...neostandard({ ignores }),
  {
    languageOptions: { ecmaVersion: 2025 }
  }
]
