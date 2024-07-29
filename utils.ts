import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const semanticSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 36,
  separators: [" "],
  chunkOverlap: 9,
});

export function splitTextIntoWords(text: string): string[] {
  return text.split(/\s/);
}

export async function splitTextIntoSemantics(text: string): Promise<string[]> {
  if (text.split(/\s/).length === 1) return []; // no semantics for single words
  const documents = await semanticSplitter.createDocuments([text]);
  const chunks = documents.map((chunk) => chunk.pageContent);
  return chunks;
}
