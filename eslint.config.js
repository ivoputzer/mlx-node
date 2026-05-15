import neostandard, { resolveIgnoresFromGitignore } from 'neostandard'
// import neostandard, { plugins, resolveIgnoresFromGitignore } from 'neostandard'

const ignores = resolveIgnoresFromGitignore()

export default [
  ...neostandard({ ignores }),
  { languageOptions: { ecmaVersion: 2025 } }
]

// export default [
//   ...neostandard({ ignores }),
//   plugins.n.configs['flat/recommended'],
//   { languageOptions: { ecmaVersion: 2025 } }
// ]
