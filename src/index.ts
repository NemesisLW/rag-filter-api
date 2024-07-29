import { Index } from "@upstash/vector";
import { Hono } from "hono";
import { env } from "hono/adapter";
import { cors } from "hono/cors";
import { splitTextIntoSemantics, splitTextIntoWords } from "../utils";

const WHITELIST = ["black", "swear", "sex"];
const FLAG_THRESHOLD = 0.78;

type ENV_CONFIG = {
  UPSTASH_VECTOR_REST_URL: string;
  UPSTASH_VECTOR_REST_TOKEN: string;
};

const app = new Hono();

app.use(cors());

app.post("/", async (ctx) => {
  if (ctx.req.header("Content-Type") !== "application/json") {
    return ctx.json({ error: "JSON expected" }, { status: 406 });
  }

  try {
    const { UPSTASH_VECTOR_REST_TOKEN, UPSTASH_VECTOR_REST_URL } =
      env<ENV_CONFIG>(ctx);

    const index = new Index({
      url: UPSTASH_VECTOR_REST_URL,
      token: UPSTASH_VECTOR_REST_TOKEN,
      //   cache: false,
    });

    const body = await ctx.req.json();
    let { message } = body as { message: string };

    if (!message) {
      return ctx.json({ error: "Message is required" }, { status: 400 });
    }

    // this is because of the cloudflare worker sub-request limit
    if (message.split(/\s/).length > 35 || message.length > 1000) {
      return ctx.json(
        {
          error:
            "Due to temporary cloudflare limits, a message can only be up to 35 words or 1000 characters.",
        },
        { status: 413 }
      );
    }

    message = message
      .split(/\s/)
      .filter((word) => !WHITELIST.includes(word.toLowerCase()))
      .join(" ");

    const [semanticChunks, wordChunks] = await Promise.all([
      splitTextIntoSemantics(message),
      splitTextIntoWords(message),
    ]);

    const flaggedFor = new Set<{ score: Number; text: string }>();

    const vectorRes = await Promise.all([
      // this additional step adds slight latency but improves output on long text massively

      ...wordChunks.map(async (wordChunk) => {
        const [vector] = await index.query({
          topK: 1,
          data: wordChunk,
          includeMetadata: true,
        });

        if (vector && vector.score > 0.95) {
          flaggedFor.add({
            text: vector.metadata!.text as string,
            score: vector.score,
          });
        }

        return { score: 0 };
      }),
      ...semanticChunks.map(async (semanticChunk) => {
        const [vector] = await index.query({
          topK: 1,
          data: semanticChunk,
          includeMetadata: true,
        });

        if (vector && vector.score > FLAG_THRESHOLD) {
          flaggedFor.add({
            text: vector.metadata!.text as string,
            score: vector.score,
          });
        }
        return vector!;
      }),
    ]);

    if (flaggedFor.size > 0) {
      const flagged = Array.from(flaggedFor).sort((a, b) =>
        a.score > b.score ? -1 : 1
      )[0];
      return ctx.json({
        isProfanity: true,
        score: flagged?.score,
        flaggedFor: flagged?.text,
      });
    } else {
      const mostFlaggedChunk = vectorRes.sort((a, b) =>
        a.score > b.score ? -1 : 1
      )[0]!;

      return ctx.json({
        isProfanity: false,
        score: mostFlaggedChunk.score,
      });
    }
  } catch (error) {
    console.error(error);

    return ctx.json(
      { error: "Something went wrong.", err: JSON.stringify(error) },
      { status: 500 }
    );
  }
});

export default app;
