// ingest.js — run ONCE to embed your PDF into Qdrant
// Usage: node ingest.js
// Prereq: docker run -p 6333:6333 qdrant/qdrant

import fs from "fs";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import { QdrantClient } from "@qdrant/js-client-rest";
import OpenAI from "openai";
import dotenv from "dotenv";
dotenv.config();

const PDF_PATH = "./javabook.pdf";
const COLLECTION = "java_book";
const BATCH_SIZE = 50; // embeddings per API call

// A chunk is considered "too big" if it exceeds this many chars.
// Oversized paragraphs/code blocks get split at this boundary.
const MAX_CHUNK_CHARS = 1200;
const MIN_CHUNK_CHARS = 60; // discard tiny fragments

// const qdrant = new QdrantClient({ url: "http://localhost:6333" });
const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
});

// ─────────────────────────────────────────────────────────────────────────────
// STEP 1 — Extract structured blocks per page
//
// Instead of one flat string we extract an array of "blocks".
// Each block has: { type: "text"|"code", text, pageNum, chapterHint }
//
// How we detect code vs prose:
//   • pdfjs gives us each text item with its x position (transform[4])
//   • Lines where most items start far from the left margin (x > leftMargin + 40)
//     are likely indented code
//   • Lines that contain Java keywords and end without a period are flagged too
//   • Consecutive code-looking lines are merged into one code block
// ─────────────────────────────────────────────────────────────────────────────
const JAVA_KEYWORDS = new Set([
  "public",
  "private",
  "protected",
  "class",
  "interface",
  "extends",
  "implements",
  "static",
  "void",
  "int",
  "long",
  "double",
  "float",
  "boolean",
  "char",
  "byte",
  "short",
  "new",
  "return",
  "if",
  "else",
  "for",
  "while",
  "do",
  "switch",
  "case",
  "break",
  "continue",
  "try",
  "catch",
  "finally",
  "throw",
  "throws",
  "import",
  "package",
  "final",
  "abstract",
  "synchronized",
  "volatile",
  "transient",
  "instanceof",
  "this",
  "super",
  "null",
  "true",
  "false",
]);

function looksLikeCode(line) {
  const trimmed = line.trim();
  // ends with { } ; or starts with typical Java patterns
  if (/[{};]$/.test(trimmed)) return true;
  if (/^\s*(\/\/|\/\*|\*)/.test(line)) return true; // comments
  if (/^\s*@\w+/.test(line)) return true; // annotations
  // contains a Java keyword followed by space/( — e.g. "public class Foo"
  const words = trimmed.split(/\s+/);
  return words.filter((w) => JAVA_KEYWORDS.has(w)).length >= 2;
}

// Detect a chapter/section heading (short line, no trailing period, all caps or title-ish)
function extractChapterHint(line) {
  const t = line.trim();
  if (t.length < 3 || t.length > 80) return null;
  // e.g. "Chapter 3: Operators" or "POLYMORPHISM" or "3.1 The if Statement"
  if (/^(chapter|part|section|\d+[\.\d]*)\s/i.test(t)) return t;
  if (t === t.toUpperCase() && /^[A-Z\s]+$/.test(t)) return t;
  return null;
}

async function extractBlocks(pdfPath) {
  console.log("Reading PDF...");
  const data = new Uint8Array(fs.readFileSync(pdfPath));
  const pdf = await pdfjsLib.getDocument({ data }).promise;
  console.log(`  ${pdf.numPages} pages`);

  const blocks = []; // final output
  const seenPages = new Set();
  let currentChapter = "Unknown";

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    if (pageNum % 200 === 0)
      console.log(`  page ${pageNum}/${pdf.numPages}...`);

    const page = await pdf.getPage(pageNum);
    const content = await page.getTextContent();

    // Deduplicate pages
    const pageSnippet = content.items
      .slice(0, 10)
      .map((i) => i.str)
      .join("");
    if (seenPages.has(pageSnippet)) continue;
    seenPages.add(pageSnippet);

    // Group items into visual lines by their Y coordinate (transform[5])
    // Items with the same Y (within 2px) belong to the same line
    const lineMap = new Map();
    for (const item of content.items) {
      if (!item.str.trim()) continue;
      const y = Math.round(item.transform[5] / 2) * 2; // bucket to 2px
      if (!lineMap.has(y)) lineMap.set(y, []);
      lineMap.get(y).push(item);
    }

    // Sort lines top-to-bottom (PDF y=0 is at the bottom, so descending)
    const sortedYs = [...lineMap.keys()].sort((a, b) => b - a);
    const lines = sortedYs.map((y) => {
      const items = lineMap
        .get(y)
        .sort((a, b) => a.transform[4] - b.transform[4]);
      const text = items
        .map((i) => i.str)
        .join(" ")
        .trim();
      const minX = Math.min(...items.map((i) => i.transform[4]));
      return { text, minX };
    });

    // Find the most common left-margin x across all lines (= body text margin)
    const xCounts = new Map();
    for (const { minX } of lines) {
      const bucket = Math.round(minX / 5) * 5;
      xCounts.set(bucket, (xCounts.get(bucket) || 0) + 1);
    }
    const bodyMarginX =
      [...xCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 50;

    // Walk lines and group into prose paragraphs or code blocks
    let currentType = null;
    let currentLines = [];

    const flushBlock = () => {
      if (!currentLines.length) return;
      const text = currentLines.join("\n").trim();
      if (text.length >= MIN_CHUNK_CHARS) {
        blocks.push({
          type: currentType,
          text,
          pageNum,
          chapterHint: currentChapter,
        });
      }
      currentLines = [];
    };

    for (const { text, minX } of lines) {
      // Update chapter tracker
      const heading = extractChapterHint(text);
      if (heading) currentChapter = heading;

      // Decide if this line is code or prose
      const isIndented = minX > bodyMarginX + 30;
      const isCode = isIndented || looksLikeCode(text);
      const lineType = isCode ? "code" : "text";

      if (lineType !== currentType) {
        flushBlock();
        currentType = lineType;
      }
      currentLines.push(text);
    }
    flushBlock();
  }

  console.log(`  extracted ${blocks.length} raw blocks`);
  return blocks;
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 2 — Smart chunking
//
// Prose blocks: split on blank-line paragraph boundaries first.
//   If a paragraph still exceeds MAX_CHUNK_CHARS, split at sentence ends.
//
// Code blocks: keep whole. Only split if truly enormous (>MAX_CHUNK_CHARS),
//   breaking at blank lines inside the code.
// ─────────────────────────────────────────────────────────────────────────────
function splitAtSentences(text, max) {
  const chunks = [];
  let current = "";
  // split on ". " or ".\n" so we break between sentences
  const sentences = text.split(/(?<=\.)\s+/);
  for (const sentence of sentences) {
    if ((current + " " + sentence).length > max && current) {
      chunks.push(current.trim());
      current = sentence;
    } else {
      current = current ? current + " " + sentence : sentence;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

function chunkBlock(block) {
  const { type, text, pageNum, chapterHint } = block;
  const results = [];

  const addChunk = (chunkText, chunkType) => {
    if (chunkText.trim().length >= MIN_CHUNK_CHARS) {
      results.push({
        type: chunkType,
        text: chunkText.trim(),
        pageNum,
        chapterHint,
      });
    }
  };

  if (type === "code") {
    if (text.length <= MAX_CHUNK_CHARS) {
      // whole code block fits → keep it intact
      addChunk(text, "code");
    } else {
      // split only on blank lines inside the code block
      const parts = text.split(/\n\s*\n/);
      let current = "";
      for (const part of parts) {
        if ((current + "\n\n" + part).length > MAX_CHUNK_CHARS && current) {
          addChunk(current, "code");
          current = part;
        } else {
          current = current ? current + "\n\n" + part : part;
        }
      }
      if (current) addChunk(current, "code");
    }
  } else {
    // prose: split on paragraph boundaries (blank lines) first
    const paragraphs = text.split(/\n\s*\n/);
    let current = "";
    for (const para of paragraphs) {
      const candidate = current ? current + "\n\n" + para : para;
      if (candidate.length > MAX_CHUNK_CHARS) {
        // flush what we have, then handle this paragraph
        if (current) {
          // current fits — chunk it
          if (current.length > MAX_CHUNK_CHARS) {
            splitAtSentences(current, MAX_CHUNK_CHARS).forEach((s) =>
              addChunk(s, "text"),
            );
          } else {
            addChunk(current, "text");
          }
          current = "";
        }
        // now deal with this paragraph alone
        if (para.length > MAX_CHUNK_CHARS) {
          splitAtSentences(para, MAX_CHUNK_CHARS).forEach((s) =>
            addChunk(s, "text"),
          );
        } else {
          current = para;
        }
      } else {
        current = candidate;
      }
    }
    if (current) {
      if (current.length > MAX_CHUNK_CHARS) {
        splitAtSentences(current, MAX_CHUNK_CHARS).forEach((s) =>
          addChunk(s, "text"),
        );
      } else {
        addChunk(current, "text");
      }
    }
  }

  return results;
}

function buildChunks(blocks) {
  const chunks = [];
  for (const block of blocks) {
    chunks.push(...chunkBlock(block));
  }
  console.log(`  ${chunks.length} final chunks (after smart splitting)`);
  return chunks;
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 3 — Embed + upsert into Qdrant with rich metadata payload
// ─────────────────────────────────────────────────────────────────────────────
async function embedBatch(texts) {
  const res = await openai.embeddings.create({
    model: "openai/text-embedding-3-small",
    input: texts,
  });
  return res.data.map((d) => d.embedding);
}

async function setupCollection() {
  const { collections } = await qdrant.getCollections();
  if (collections.some((c) => c.name === COLLECTION)) {
    console.log(`Collection "${COLLECTION}" exists — recreating...`);
    await qdrant.deleteCollection(COLLECTION);
  }
  await qdrant.createCollection(COLLECTION, {
    vectors: { size: 1536, distance: "Cosine" },
  });
  console.log(`Collection "${COLLECTION}" ready`);
}

async function upsertChunks(chunks) {
  console.log(
    `\nEmbedding ${chunks.length} chunks in batches of ${BATCH_SIZE}...`,
  );
  let done = 0;

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const embeddings = await embedBatch(batch.map((c) => c.text));

    const points = batch.map((chunk, j) => ({
      id: i + j,
      vector: embeddings[j],
      payload: {
        text: chunk.text, // raw content for context injection
        type: chunk.type, // "text" | "code"
        page: chunk.pageNum, // page number in the original PDF
        chapter: chunk.chapterHint, // nearest heading seen before this chunk
        char_count: chunk.text.length, // useful for debugging chunk sizes
        chunk_index: i + j, // global position in the book
      },
    }));

    await qdrant.upsert(COLLECTION, { wait: true, points });
    done += batch.length;
    console.log(`  ${done}/${chunks.length}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
async function ingest() {
  const blocks = await extractBlocks(PDF_PATH);
  const chunks = buildChunks(blocks);

  const codeCount = chunks.filter((c) => c.type === "code").length;
  const textCount = chunks.filter((c) => c.type === "text").length;
  console.log(`  ${textCount} prose chunks, ${codeCount} code chunks`);

  await setupCollection();
  await upsertChunks(chunks);

  console.log(`\nDone! ${chunks.length} chunks stored in Qdrant.`);
  console.log(
    `Each chunk has: text, type, page, chapter, char_count, chunk_index`,
  );
}

ingest().catch(console.error);
