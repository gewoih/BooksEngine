import { ActionIcon, Group, Rating, Tooltip } from "@mantine/core";
import { useMediaQuery } from "@mantine/hooks";
import type { Book } from "../api/client";
import { useRate } from "../api/hooks";
import classes from "./RatingControl.module.css";

/** Звёзды 1–5 и «бросил». Проявляются при наведении на строку; на сенсорном экране и у оценённых — видны всегда. */
export default function RatingControl({ book, alwaysVisible }: { book: Book; alwaysVisible?: boolean }) {
  const rate = useRate();
  const touch = useMediaQuery("(hover: none), (max-width: 48em)");
  const rated = book.myRating != null;
  return (
    <Group gap={6} className={alwaysVisible || touch || rated ? undefined : classes.onHover} wrap="nowrap">
      <Rating value={book.myDnf ? 0 : book.myRating ?? 0}
              onChange={(v) => rate.mutate({ workId: book.workId, value: v, book })} />
      <Tooltip label={book.myDnf ? "Бросил (= 1★). Нажмите, чтобы убрать" : "Бросил, не дочитал (= 1★)"}>
        <ActionIcon variant={book.myDnf ? "filled" : "subtle"} color="gray" size="sm" aria-label="Бросил"
                    onClick={() => rate.mutate({ workId: book.workId, dnf: !book.myDnf, remove: book.myDnf, book })}>
          ✕
        </ActionIcon>
      </Tooltip>
      {rated && (
        <Tooltip label="Удалить оценку">
          <ActionIcon variant="subtle" color="red" size="sm" aria-label="Удалить оценку"
                      onClick={() => rate.mutate({ workId: book.workId, remove: true, book })}>🗑</ActionIcon>
        </Tooltip>
      )}
    </Group>
  );
}
