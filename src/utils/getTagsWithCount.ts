import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

export interface TagWithCount {
  tag: string;
  tagName: string;
  count: number;
}

const getTagsWithCount = (
  posts: CollectionEntry<"blog">[]
): TagWithCount[] => {
  const tagCountMap = new Map<string, { tagName: string; count: number }>();

  posts.filter(postFilter).forEach(post => {
    post.data.tags.forEach(rawTag => {
      const slug = slugifyStr(rawTag);
      const existing = tagCountMap.get(slug);
      if (existing) {
        existing.count++;
      } else {
        tagCountMap.set(slug, { tagName: rawTag, count: 1 });
      }
    });
  });

  return Array.from(tagCountMap.entries())
    .map(([tag, { tagName, count }]) => ({ tag, tagName, count }))
    .sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
};

export default getTagsWithCount;
