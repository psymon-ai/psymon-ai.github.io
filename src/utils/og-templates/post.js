import satori from "satori";
import fs from "fs";
import path from "path";
import { SITE } from "@/config";

const regularFont = fs.readFileSync(
  path.resolve("./src/assets/fonts/Pretendard-Regular.otf")
);

const boldFont = fs.readFileSync(
  path.resolve("./src/assets/fonts/Pretendard-Bold.otf")
);

export default async post => {
  const title = post.data.title;
  const author = post.data.author;
  const tags = post.data.tags ?? [];

  return satori(
    {
      type: "div",
      props: {
        style: {
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          background:
            "linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%)",
          padding: "80px",
          color: "#fff",
          fontFamily: "Pretendard",
        },

        children: [
          {
            type: "div",
            props: {
              style: {
                fontSize: 34,
                opacity: 0.9,
              },
              children: SITE.title,
            },
          },

          {
            type: "div",
            props: {
              style: {
                fontSize: 72,
                fontWeight: 700,
                lineHeight: 1.2,
                maxHeight: "70%",
                overflow: "hidden",
                wordBreak: "keep-all",
              },
              children: title,
            },
          },

          {
            type: "div",
            props: {
              style: {
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                fontSize: 30,
                opacity: 0.9,
              },

              children: [
                {
                  type: "div",
                  props: {
                    children: `by ${author}`,
                  },
                },

                {
                  type: "div",
                  props: {
                    style: {
                      display: "flex",
                      gap: "12px",
                    },

                    children: tags.slice(0, 3).map(tag => ({
                      type: "span",
                      props: {
                        style: {
                          background: "rgba(255,255,255,0.15)",
                          padding: "6px 14px",
                          borderRadius: "999px",
                          fontSize: 24,
                        },
                        children: tag,
                      },
                    })),
                  },
                },
              ],
            },
          },
        ],
      },
    },

    {
      width: 1200,
      height: 630,
      fonts: [
        {
          name: "Pretendard",
          data: regularFont,
          weight: 400,
        },
        {
          name: "Pretendard",
          data: boldFont,
          weight: 700,
        },
      ],
    }
  );
};