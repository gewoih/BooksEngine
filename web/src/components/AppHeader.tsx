import { Button, Group, Text } from "@mantine/core";
import { NavLink } from "react-router";
import { useLogout, useSession } from "../api/hooks";
import ImportModal from "./ImportModal";

export default function AppHeader() {
  const session = useSession();
  const logout = useLogout();
  return (
    <Group h="100%" px="md" justify="space-between" wrap="nowrap">
      <Group gap="xs" wrap="nowrap">
        <Text fw={700} visibleFrom="sm">BooksEngine</Text>
        <Button component={NavLink} to="/" variant="subtle" size="compact-md">Библиотека</Button>
        <Button component={NavLink} to="/recommendations" variant="subtle" size="compact-md">Рекомендации</Button>
      </Group>
      <Group gap="xs" wrap="nowrap">
        <ImportModal />
        <Text size="sm" visibleFrom="xs">{session.data?.name}</Text>
        <Button variant="default" size="xs" onClick={() => logout.mutate()}>Сменить</Button>
      </Group>
    </Group>
  );
}
