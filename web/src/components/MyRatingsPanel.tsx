import { Button, Collapse, Divider, Paper, Stack, Text } from "@mantine/core";
import { useLocalStorage } from "@mantine/hooks";
import { useMyRatings } from "../api/hooks";
import BookRow from "./BookRow";

export default function MyRatingsPanel() {
  const ratings = useMyRatings();
  const [open, setOpen] = useLocalStorage({ key: "my-ratings-open", defaultValue: true });
  const items = ratings.data ?? [];
  return (
    <Paper withBorder p="sm" mb="md">
      <Button variant="subtle" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} Мои оценки ({items.length})
      </Button>
      <Collapse expanded={open}>
        {items.length === 0
          ? <Text c="dimmed" size="sm" p="sm">Пока пусто — оцените книги ниже или импортируйте CSV.</Text>
          : <Stack gap={0} mah={420} style={{ overflowY: "auto" }}>
              {items.map((r) => (<div key={r.book.workId}><BookRow book={r.book} ratingAlwaysVisible /><Divider /></div>))}
            </Stack>}
      </Collapse>
    </Paper>
  );
}
