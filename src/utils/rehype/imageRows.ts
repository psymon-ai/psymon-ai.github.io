import type { Root, Element, Comment } from "hast";
import type { Plugin } from "unified";

/**
 * Wraps consecutive image paragraphs between:
 * <!-- img-row-2:start --> ... <!-- img-row:end -->
 * or
 * <!-- img-row-3:start --> ... <!-- img-row:end -->
 *
 * into:
 * <div class="img-row-2">...</div>
 * / <div class="img-row-3">...</div>
 *
 * The inner image markup itself stays untouched so Astro's
 * content image handling continues to work.
 */
const rehypeImageRows: Plugin<[], Root> = () => {
  return (tree: Root) => {
    const children = tree.children;

    let i = 0;
    while (i < children.length) {
      const node = children[i];

      if (node.type === "comment") {
        const comment = (node as Comment).value.trim();

        const isRow2 = comment.startsWith("img-row-2:start");
        const isRow3 = comment.startsWith("img-row-3:start");

        if (!isRow2 && !isRow3) {
          i++;
          continue;
        }

        const className = isRow2 ? "img-row-2" : "img-row-3";

        // Collect nodes until end marker
        const collected: Root["children"] = [];
        let j = i + 1;
        let endIndex = -1;

        while (j < children.length) {
          const next = children[j];

          if (next.type === "comment") {
            const endComment = (next as Comment).value.trim();
            if (endComment.startsWith("img-row:end")) {
              endIndex = j;
              break;
            }
          }

          collected.push(next);
          j++;
        }

        if (endIndex === -1 || collected.length === 0) {
          // No valid end marker or no content; skip
          i++;
          continue;
        }

        const wrapper: Element = {
          type: "element",
          tagName: "div",
          properties: { className: [className] },
          children: collected as any,
        };

        // Replace from start marker to end marker (inclusive)
        children.splice(i, endIndex - i + 1, wrapper);

        // Continue after the newly inserted wrapper
        i++;
        continue;
      }

      i++;
    }
  };
};

export default rehypeImageRows;

