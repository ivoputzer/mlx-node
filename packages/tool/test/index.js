import { loadTemplate, loadTokenizer } from 'mlx-lm'
import { join } from 'path'
import { analyzeToolMarkers } from '../index.js'
import { readdir } from 'fs/promises'

(await readdir(join(import.meta.dirname, 'models'), { withFileTypes: true }))
  .filter(dirent => dirent.isDirectory())
  .map(dirent => dirent.name)
  .forEach(async model => {
    const path = join(import.meta.dirname, 'models', model)
    const tokenizer = await loadTokenizer(path)
    const template = await loadTemplate(path, tokenizer)

    console.log('\n--- %s ---\n%s\n===', path, true || template.format({ indent: 2 }))
    console.log('analyzeTemplateMarkers:', analyzeToolMarkers(template, tokenizer), '\n===')
  })
