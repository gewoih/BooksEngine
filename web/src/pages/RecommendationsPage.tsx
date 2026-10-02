import { Alert, Badge, Center, Divider, Group, Loader, SegmentedControl, Spoiler, Stack, Tabs, Text, Tooltip } from "@mantine/core";
import { useLocalStorage } from "@mantine/hooks";
import { useRecommendations } from "../api/hooks";
import type { Recommendation } from "../api/client";
import BookRow from "../components/BookRow";

/** Сдвиг книги после последнего изменения оценок: ▲/▼ с подсказкой, «новая» — раньше её в списке не было. */
function Move({ r }: { r: Recommendation }) {
  if (r.isNew) {
    return (
      <Tooltip label="До изменения оценок этой книги в списке не было (ниже 50-го места)">
        <Badge size="sm" color="teal" variant="light">новая</Badge>
      </Tooltip>
    );
  }
  if (r.previousRank == null || r.previousRank === r.rank) return null;
  const up = r.previousRank > r.rank;
  return (
    <Tooltip label={`Было ${r.previousRank}-е место → стало ${r.rank}-е`}>
      <Text size="sm" fw={700} c={up ? "green" : "red"} style={{ cursor: "default" }}>
        {up ? "▲" : "▼"} {Math.abs(r.previousRank - r.rank)}
      </Text>
    </Tooltip>
  );
}

function chanceColor(pct: number) {
  return pct >= 45 ? "green" : pct >= 30 ? "yellow" : "gray";
}

export default function RecommendationsPage() {
  const [n, setN] = useLocalStorage({ key: "recs-n", defaultValue: "20" });
  const [tab, setTab] = useLocalStorage<string | null>({ key: "recs-tab", defaultValue: null });
  const recs = useRecommendations(Number(n));
  const data = recs.data;
  const sections = data?.sections ?? [];
  const active = sections.some((s) => s.name === tab) ? tab : sections[0]?.name ?? null;
  return (
    <>
      <Group justify="space-between" mb="md">
        <Text c="dimmed" size="sm">{data && data.usedRatings ? `По ${data.usedRatings} вашим оценкам` : ""}</Text>
        <SegmentedControl value={n} onChange={setN} data={["10", "20", "50"]} />
      </Group>
      {recs.isLoading && <Center><Loader /></Center>}
      {recs.error && <Alert color="red" title="Рекомендации недоступны">{recs.error.message}</Alert>}
      {data && sections.length === 0 && (
        <Alert title="Пока нечего советовать">Оцените книги в библиотеке или импортируйте CSV.</Alert>
      )}
      {data && sections.length > 0 && (
        <Stack gap="xs" mb="sm">
          <Text size="sm" c="dimmed">
            В значке — шанс, что поставите книге {data.chanceLabel}, если прочтёте (из десяти книг с шансом 40%
            {" "}{data.chanceLabel} получат четыре).
          </Text>
          <Spoiler maxHeight={0} showLabel="Как читать подписи" hideLabel="Скрыть">
            <Text size="sm" c="dimmed">{data.legend}</Text>
          </Spoiler>
          {data.changes.length > 0 && (
            <Text size="sm" c="dimmed">
              Стрелки — как сдвинулись книги после последнего изменения оценок ({data.changes.join("; ")}).
            </Text>
          )}
        </Stack>
      )}
      {sections.length > 0 && (
        <Tabs value={active} onChange={setTab}>
          <Tabs.List mb="xs">
            {sections.map((s) => <Tabs.Tab key={s.name} value={s.name}>{s.name}</Tabs.Tab>)}
          </Tabs.List>
          {sections.map((s) => (
            <Tabs.Panel key={s.name} value={s.name}>
              {s.note && <Text size="sm" c="dimmed" mb="xs">{s.note}</Text>}
              <Stack gap={0}>
                {s.items.map((r) => (
                  <div key={r.book.workId}>
                    <BookRow book={r.book} extra={
                      <Stack gap={2}>
                        <Group gap="xs">
                          <Text size="sm" c="dimmed">#{r.rank}</Text>
                          <Badge color={chanceColor(r.chancePct)}>{data!.chanceLabel}: {r.chancePct}%</Badge>
                          <Move r={r} />
                        </Group>
                        {r.why.map((w) => <Text key={w} size="sm">{w}</Text>)}
                      </Stack>
                    } />
                    <Divider />
                  </div>
                ))}
              </Stack>
            </Tabs.Panel>
          ))}
        </Tabs>
      )}
    </>
  );
}
