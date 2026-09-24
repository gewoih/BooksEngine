import { Anchor, Badge, Group, Stack, Text } from "@mantine/core";
import type { ReactNode } from "react";
import { useSearchParams } from "react-router";
import type { Book } from "../api/client";
import Cover from "./Cover";
import RatingControl from "./RatingControl";

export default function BookRow({ book, extra, ratingAlwaysVisible }:
  { book: Book; extra?: ReactNode; ratingAlwaysVisible?: boolean }) {
  const [params, setParams] = useSearchParams();
  const open = () => { params.set("work", String(book.workId)); setParams(params); };
  return (
    <Group className="book-row" wrap="nowrap" align="flex-start" py={6}>
      <Cover url={book.coverUrl} title={book.title} />
      <Stack gap={2} style={{ flex: 1, minWidth: 0 }}>
        <Anchor component="button" onClick={open} fw={600} ta="left" lineClamp={2}>{book.title}</Anchor>
        <Text size="sm" c="dimmed">{[book.author, book.year].filter(Boolean).join(", ")}</Text>
        {!book.inCore && <Badge size="xs" color="gray">не влияет на рекомендации</Badge>}
        {extra}
      </Stack>
      <RatingControl book={book} alwaysVisible={ratingAlwaysVisible} />
    </Group>
  );
}
