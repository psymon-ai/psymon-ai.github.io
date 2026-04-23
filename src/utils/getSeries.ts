import type { CollectionEntry } from "astro:content";
import { BLOG_PATH } from "@/content.config";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

export interface Series {
  /** Slugified series name (used in URLs) */
  slug: string;
  /** Original folder name (display name) */
  name: string;
  /** Number of published posts in this series */
  postCount: number;
  /** Most recent publish/modify date among posts in this series */
  latestDate: Date;
}

/**
 * Extract the series (subfolder) name from a post's filePath.
 * Returns undefined if the post is at the root of blog directory (no series).
 */
export function getSeriesFromPost(
  filePath: string | undefined
): { slug: string; name: string } | undefined {
  if (!filePath) return undefined;

  const relativePath = filePath.replace(BLOG_PATH, "");
  const segments = relativePath
    .split("/")
    .filter(s => s !== "")
    .filter(s => !s.startsWith("_"));

  // If there are 2+ segments, the first one(s) before the filename are the series folder
  // e.g. "AI 개발을 위한 배경 지식/기초이론 1.md" → series = "AI 개발을 위한 배경 지식"
  if (segments.length >= 2) {
    const seriesName = segments[0];
    return {
      slug: slugifyStr(seriesName),
      name: seriesName,
    };
  }

  return undefined;
}

/**
 * Get all unique series from blog posts with metadata.
 */
export function getUniqueSeries(
  posts: CollectionEntry<"blog">[]
): Series[] {
  const seriesMap = new Map<
    string,
    { name: string; postCount: number; latestDate: Date }
  >();

  posts.filter(postFilter).forEach(post => {
    const series = getSeriesFromPost(post.filePath);
    if (!series) return;

    const existing = seriesMap.get(series.slug);
    const postDate = new Date(
      post.data.modDatetime ?? post.data.pubDatetime
    );

    if (existing) {
      existing.postCount++;
      if (postDate > existing.latestDate) {
        existing.latestDate = postDate;
      }
    } else {
      seriesMap.set(series.slug, {
        name: series.name,
        postCount: 1,
        latestDate: postDate,
      });
    }
  });

  return Array.from(seriesMap.entries())
    .map(([slug, data]) => ({
      slug,
      name: data.name,
      postCount: data.postCount,
      latestDate: data.latestDate,
    }))
    .sort((a, b) => b.latestDate.getTime() - a.latestDate.getTime());
}

/**
 * Get all published posts belonging to a specific series slug.
 * Returns posts sorted by pubDatetime ascending (chronological order within series).
 */
export function getPostsBySeries(
  posts: CollectionEntry<"blog">[],
  seriesSlug: string
): CollectionEntry<"blog">[] {
  return posts
    .filter(postFilter)
    .filter(post => {
      const series = getSeriesFromPost(post.filePath);
      return series?.slug === seriesSlug;
    })
    .sort(
      (a, b) =>
        new Date(a.data.pubDatetime).getTime() -
        new Date(b.data.pubDatetime).getTime()
    );
}
