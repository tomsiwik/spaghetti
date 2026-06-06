import { initFts } from "./packages/db/src/db";

console.log("Initializing FTS index...");
await initFts();
console.log("FTS index initialized successfully.");
process.exit(0);
