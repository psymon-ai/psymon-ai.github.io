import { defineConfig, envField, fontProviders } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import sitemap from "@astrojs/sitemap";
import remarkToc from "remark-toc";
import remarkCollapse from "remark-collapse";
import { SITE } from "./src/config";
import rehypeExternalLinks from "rehype-external-links";
import expressiveCode from 'astro-expressive-code';
import { pluginLineNumbers } from '@expressive-code/plugin-line-numbers';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeImageCaption from './src/utils/rehype/imageCaption';
import { remarkFixCjkBold } from './src/utils/remark/remarkFixCjkBold';

// https://astro.build/config
export default defineConfig({
  site: SITE.website,
  base: "/",
  trailingSlash: "ignore",
  integrations: [
    sitemap({
      filter: page => SITE.showArchives || !page.endsWith("/archives"),
    }),
    expressiveCode({
      // 라인 넘버 활성화
      plugins: [
        pluginLineNumbers(),
      ],

      // 기본 props: wrap + line numbers 기본 ON
      defaultProps: {
        showLineNumbers: true,         // fence에 showLineNumbers 안 써도 기본으로 나옴
        wrap: true,
        preserveIndent: true,
        overridesByLang: {
          'bash,sh,zsh,shell,console': {
            frame: 'code',      // 또는 'none'으로 프레임 자체 제거
            // 필요하면 showLineNumbers: false 등도 추가 가능
          },
        },
      },

      // 스타일 커스텀
      styleOverrides: {
        codePaddingBlock: '2.5rem', 
        borderRadius: '0.8rem',
        codeFontSize: '0.92rem',
        lineNumbers: {
          foreground: '#6e7681',          // VS Code 느낌 회색
          highlightForeground: '#c9d1d9', // 강조 시 밝게
        },
      },

      frames: true,

      themes: ['github-dark-high-contrast'], // 또는 dracula 등
    }),
  ],
  markdown: {
    gfm: true,
    remarkPlugins: [
      remarkToc,
      remarkMath,
      remarkFixCjkBold,
      [remarkCollapse, { test: "Table of contents" }],
    ],
    rehypePlugins: [
      rehypeKatex,
      rehypeImageCaption,
      [rehypeExternalLinks, { target: "_blank", rel: ["noopener", "noreferrer"] }],
    ],
    // syntaxHighlight: "shiki", // Shiki 대신 Expressive Code로 대체
    // shikiConfig: {
    //   // For more themes, visit https://shiki.style/themes
    //   theme: "github-dark-high-contrast",
    //   defaultColor: false,
    //   wrap: false,
    //   transformers: [
    //     transformerFileName({ style: "v2", hideDot: false }),
    //     transformerNotationHighlight(),
    //     transformerNotationWordHighlight(),
    //     transformerNotationDiff({ matchAlgorithm: "v3" }),
    //   ],
    // },
  },
  vite: {
    // eslint-disable-next-line
    // @ts-ignore
    // This will be fixed in Astro 6 with Vite 7 support
    // See: https://github.com/withastro/astro/issues/14030
    plugins: [tailwindcss()],
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"],
    },
  },
  image: {
    responsiveStyles: true,
    layout: "constrained",
  },
  env: {
    schema: {
      PUBLIC_GOOGLE_SITE_VERIFICATION: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
      PUBLIC_GA_MEASUREMENT_ID: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
      PUBLIC_UMAMI_HOST: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
      PUBLIC_UMAMI_WEBSITE_ID: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
      UMAMI_API_KEY: envField.string({
        access: "secret",
        context: "server",
        optional: true,
      }),
    },
  },
  experimental: {
    preserveScriptOrder: true,
    fonts: [
      {
        name: "Hahmlet",
        cssVariable: "--font-hahmlet",
        provider: fontProviders.google(),
        weights: [400, 500, 600, 700, 800, 900],
        styles: ["normal"],
      },
      {
        name: "IBM Plex Sans",
        cssVariable: "--font-ibm-plex-sans",
        provider: fontProviders.google(),
        weights: [300, 400, 500, 600, 700],
        styles: ["normal", "italic"],
      },
      {
        name: "Google Sans Code",
        cssVariable: "--font-google-sans-code",
        provider: fontProviders.google(),
        fallbacks: ["monospace"],
        weights: [300, 400, 500, 600, 700],
        styles: ["normal", "italic"],
      },
    ],
  },
});
