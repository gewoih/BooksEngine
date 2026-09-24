import { Badge, Divider, Drawer, Group, Loader, Stack, Text } from "@mantine/core";
import { useSearchParams } from "react-router";
import { useWork } from "../api/hooks";
import BookRow from "./BookRow";
import Cover from "./Cover";
import RatingControl from "./RatingControl";

/** Карточка книги поверх любой страницы: открыта, пока в адресе есть ?work=<id> («назад» её закрывает). */
export default function WorkDrawer() {
  const [params, setParams] = useSearchParams();
  const id = params.get("work") ? Number(params.get("work")) : null;
  const work = useWork(id);
  const close = () => { params.delete("work"); setParams(params); };
  const w = work.data;
  return (
    <Drawer opened={id != null} onClose={close} position="right" size="lg" title={w?.book.title}>
      {work.isLoading && <Loader />}
      {work.error && <Text c="red">{work.error.message}</Text>}
      {w && (
        <Stack>
          <Group align="flex-start" wrap="nowrap">
            <Cover url={w.largeCoverUrl} size="l" title={w.book.title} />
            <Stack gap={4}>
              <Text>{w.authors.join(", ")}</Text>
              {w.book.year && <Text c="dimmed">{w.book.year}</Text>}
              <Text size="sm" c="dimmed">оценок в Goodreads-ядре: {w.book.cfRatings}</Text>
              {w.chancePct != null && <Badge size="lg">понравится: {w.chancePct}%</Badge>}
              {w.chancePct == null && w.book.myRating == null && !w.inModel && <Badge color="gray">мало данных</Badge>}
              <RatingControl book={w.book} alwaysVisible />
            </Stack>
          </Group>
          <Group gap={4}>{w.genres.slice(0, 8).map((g) => <Badge key={g} variant="light">{g}</Badge>)}</Group>
          {w.description && <Text size="sm" style={{ whiteSpace: "pre-line" }}>{w.description}</Text>}
          {w.similar.length > 0 && (
            <>
              <Divider label="Похожие" />
              {w.similar.map((b) => <BookRow key={b.workId} book={b} />)}
            </>
          )}
        </Stack>
      )}
    </Drawer>
  );
}
