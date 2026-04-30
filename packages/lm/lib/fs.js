import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

export function expand (reltive, { homedir } = os, { join, resolve } = path) {
  if (reltive[0] === '~') {
    return join(homedir(), reltive.slice(1))
  }
  return resolve(reltive)
}

export async function isFile (path, { access, constants } = fs) {
  try {
    await access(path, constants.F_OK)
    return true
  } catch (err) {
    return false
  }
}
