import { join } from 'node:path'
import fs from 'node:fs'
import * as jinja from '@huggingface/jinja'

export async function loadTemplate (base, tokenizer, { Template } = jinja, { readFile } = fs) {
  const config = tokenizer.config
  try {
    const path = join(base, 'chat_template.jinja')
    return new Template(await readFile(path, 'utf8'))
  } catch {
    if (!config?.chat_template?.length) {
      throw new Error('No valid template found in model path.')
    }
    if (Array.isArray(config.chat_template)) {
      const templates = config.chat_template.reduce((templates, { name, template }) => ({ ...templates, [name]: new Template(template) }), {})
      return {
        render (context) {
          try {
            return context?.tools?.length && templates?.tool_use
              ? templates?.tool_use.render(context)
              : templates?.default.render(context)
          } catch {
            throw new Error('No valid template found in tokenizer config.')
          }
        }
      }
    } else {
      return new Template(config.chat_template)
    }
  }
}
