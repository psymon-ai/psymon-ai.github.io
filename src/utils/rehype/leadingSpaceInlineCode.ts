import type { Element, Root, Text } from "hast";
import type { Plugin } from "unified";

function isElement(node: unknown): node is Element {
  return (
    typeof node === "object" &&
    node !== null &&
    "type" in node &&
    node.type === "element"
  );
}

function isText(node: unknown): node is Text {
  return (
    typeof node === "object" &&
    node !== null &&
    "type" in node &&
    node.type === "text"
  );
}

function getClasses(node: Element) {
  const current = node.properties?.className;

  if (Array.isArray(current)) {
    return current.map(String);
  }

  if (typeof current === "string") {
    return current.split(/\s+/).filter(Boolean);
  }

  return [];
}

function addClass(node: Element, className: string) {
  const classes = getClasses(node);

  if (!classes.includes(className)) {
    classes.push(className);
  }

  node.properties = {
    ...node.properties,
    className: classes,
  };
}

const rehypeLeadingSpaceInlineCode: Plugin<[], Root> = () => {
  function visit(node: unknown, insidePre = false) {
    if (!isElement(node)) {
      return;
    }

    const nextInsidePre = insidePre || node.tagName === "pre";

    if (!nextInsidePre && node.tagName === "code") {
      const first = node.children[0];

      if (isText(first)) {
        const match = first.value.match(/^[ \t]+/);

        if (match) {
          const leading = match[0];
          const rest = first.value.slice(leading.length);

          addClass(node, "code-with-leading-space");

          node.children.splice(
            0,
            1,
            ...leading.split("").map(
              (space): Element => ({
                type: "element",
                tagName: "span",
                properties: { className: ["code-leading-space"] },
                children: [{ type: "text", value: space }],
              })
            ),
            ...(rest ? [{ type: "text", value: rest } as Text] : [])
          );

          return;
        }
      }
    }

    for (const child of node.children ?? []) {
      visit(child, nextInsidePre);
    }
  }

  return (tree: Root) => {
    for (const child of tree.children) {
      visit(child);
    }
  };
};

export default rehypeLeadingSpaceInlineCode;
