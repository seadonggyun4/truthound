// Parse native diagrams in a real Mermaid runtime; no remote scripts are loaded.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { JSDOM } from 'jsdom';
const mermaidPackage = JSON.parse(fs.readFileSync(new URL('./node_modules/mermaid/package.json', import.meta.url), 'utf8'));

const option = (name) => {
  const index = process.argv.indexOf(name);
  if (index < 0 || !process.argv[index + 1]) throw new Error(`Missing ${name}`);
  return process.argv[index + 1];
};
const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Element = dom.window.Element;
const { default: mermaid } = await import('mermaid');
mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });
const cases = JSON.parse(fs.readFileSync(option('--input'), 'utf8'));
assert.equal(cases.length, 5);
for (const diagram of cases) {
  const parsed = await mermaid.parse(diagram);
  assert.equal(parsed.diagramType, 'flowchart-v2');
  const native = await mermaid.mermaidAPI.getDiagramFromText(diagram);
  assert.equal(native.db.getVertices().size, 4);
  assert.equal(native.db.getEdges().length, 3);
}
const report = { status: 'passed', mermaid: mermaidPackage.version, diagrams: cases.length,
  actual_parser: true, distinct_vertices_per_diagram: 4, edges_per_diagram: 3 };
fs.writeFileSync(option('--output'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
