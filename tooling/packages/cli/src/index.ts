#!/usr/bin/env bun
import { execute } from "@oclif/core";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";

// oclif needs the directory of the CLI's package.json to discover commands
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

await execute({ development: true, dir: resolve(__dirname, "..") });
