import neostandard, { plugins, resolveIgnoresFromGitignore } from 'neostandard'

const ignores = resolveIgnoresFromGitignore()
const node = plugins.n.configs['flat/recommended']

export default [
  ...neostandard({ ignores }),
  node,
  { languageOptions: { ecmaVersion: 2025 } }
]
