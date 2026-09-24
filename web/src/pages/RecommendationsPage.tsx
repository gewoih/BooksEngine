import { Alert, Badge, Center, Divider, Group, Loader, SegmentedControl, Stack, Text } from "@mantine/core";
import { useLocalStorage } from "@mantine/hooks";
import { useRecommendations } from "../api/hooks";
import BookRow from "../components/BookRow";

export default function RecommendationsPage() {
  const [n, setN] = useLocalStorage({ key: "recs-n", defaultValue: "20" });
  const recs = useRecommendations(Number(n));
  return (
    <>
      <Group justify="space-between" mb="md">
        <Text c="dimmed" size="sm">{recs.data ? `По ${recs.data.usedRatings} вашим оценкам` : ""}</Text>
        <SegmentedControl value={n} onChange={setN} data={["10", "20", "50"]} />
      </Group>
      {recs.isLoading && <Center><Loader /></Center>}
      {recs.error && <Alert color="red" title="Рекомендации недоступны">{recs.error.message}</Alert>}
      {recs.data && recs.data.items.length === 0 && (
        <Alert title="Пока нечего советовать">Оцените книги в библиотеке или импортируйте CSV.</Alert>
      )}
      <Stack gap={0}>
        {recs.data?.items.map((r, i) => (
          <div key={r.book.workId}>
            <BookRow book={r.book} extra={
              <Stack gap={2}>
                <Group gap="xs">
                  <Text size="sm" c="dimmed">#{i + 1}</Text>
                  <Badge color={r.chancePct >= 70 ? "green" : r.chancePct >= 50 ? "yellow" : "gray"}>
                    понравится: {r.chancePct}%
                  </Badge>
                </Group>
                <Text size="sm">
                  {r.because.length ? `потому что вы оценили: ${r.because.map((b) => b.title).join(", ")}` : "по общему вкусу"}
                  {r.despite && `; несмотря на: ${r.despite.title}`}
                </Text>
              </Stack>
            } />
            <Divider />
          </div>
        ))}
      </Stack>
    </>
  );
}
