import type { Root, Element } from "hast";
import type { Plugin } from "unified";

/**
 * rehype-image-caption
 *
 * 이미지 + 이탤릭 캡션을 <figure> + <figcaption>으로 변환합니다.
 *
 * 지원 패턴:
 *
 * 패턴 A — 같은 <p> 안에 img/picture + em (줄바꿈만 있고 빈 줄 없을 때):
 *   ![alt](img.webp)
 *   *캡션*
 *
 * 패턴 B — 별도 <p>에 img/picture와 em이 각각 있을 때 (빈 줄로 구분):
 *   ![alt](img.webp)
 *
 *   *캡션*
 */

function isElement(node: any): node is Element {
  return node?.type === "element";
}

function hasImage(node: Element): boolean {
  return node.children.some(
    (c: any) =>
      isElement(c) &&
      (c.tagName === "img" || c.tagName === "picture")
  );
}

function findImages(node: Element): Element[] {
  return node.children.filter(
    (c: any) =>
      isElement(c) &&
      (c.tagName === "img" || c.tagName === "picture")
  ) as Element[];
}

function findEms(node: Element): Element[] {
  return node.children.filter(
    (c: any) => isElement(c) && c.tagName === "em"
  ) as Element[];
}

/** <p> 안에 이미지만 있는지 (whitespace 텍스트 제외) */
function isImageOnlyParagraph(node: any): boolean {
  if (!isElement(node) || node.tagName !== "p") return false;
  const meaningful = node.children.filter(
    (c: any) => !(c.type === "text" && c.value.trim() === "")
  );
  return (
    meaningful.length >= 1 &&
    meaningful.every(
      (c: any) =>
        isElement(c) && (c.tagName === "img" || c.tagName === "picture")
    )
  );
}

/** <p> 안에 <em>만 있는지 */
function isCaptionOnlyParagraph(node: any): boolean {
  if (!isElement(node) || node.tagName !== "p") return false;
  const meaningful = node.children.filter(
    (c: any) => !(c.type === "text" && c.value.trim() === "")
  );
  return (
    meaningful.length === 1 &&
    isElement(meaningful[0]) &&
    meaningful[0].tagName === "em"
  );
}

const rehypeImageCaption: Plugin<[], Root> = () => {
  return (tree: Root) => {
    const children = tree.children;
    let i = 0;

    while (i < children.length) {
      const node = children[i];

      // ── 패턴 A: <p> 안에 img/picture + em이 함께 있는 경우 ──
      if (isElement(node) && node.tagName === "p" && hasImage(node)) {
        const ems = findEms(node);
        const imgs = findImages(node);

        if (ems.length > 0 && imgs.length > 0) {
          // 마지막 <em>을 캡션으로 사용
          const captionEm = ems[ems.length - 1];

          // <p>에서 캡션 <em>과 그 앞의 줄바꿈 제거
          const emIndex = node.children.indexOf(captionEm);
          // em 앞의 줄바꿈 텍스트 노드도 제거
          let removeFrom = emIndex;
          if (
            removeFrom > 0 &&
            node.children[removeFrom - 1].type === "text" &&
            (node.children[removeFrom - 1] as any).value.trim() === ""
          ) {
            removeFrom--;
          }
          node.children.splice(removeFrom, emIndex - removeFrom + 1);

          const figure: Element = {
            type: "element",
            tagName: "figure",
            properties: { className: ["image-with-caption"] },
            children: [
              // 이미지를 포함한 원래 <p>를 그대로 넣음 (Astro 이미지 처리 유지)
              node,
              {
                type: "element",
                tagName: "figcaption",
                properties: {},
                children: captionEm.children,
              },
            ],
          };

          children.splice(i, 1, figure);
          i++;
          continue;
        }
      }

      // ── 패턴 B: 이미지 <p> 다음에 캡션 <p>가 오는 경우 ──
      if (
        i < children.length - 1 &&
        isImageOnlyParagraph(node) &&
        isCaptionOnlyParagraph(children[i + 1])
      ) {
        const next = children[i + 1] as Element;
        const captionEm = findEms(next)[0];

        const figure: Element = {
          type: "element",
          tagName: "figure",
          properties: { className: ["image-with-caption"] },
          children: [
            node as Element,
            {
              type: "element",
              tagName: "figcaption",
              properties: {},
              children: captionEm.children,
            },
          ],
        };

        children.splice(i, 2, figure);
        i++;
        continue;
      }

      i++;
    }
  };
};

export default rehypeImageCaption;
