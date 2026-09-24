import { Button, Center, Loader, Paper, Select, Stack, TextInput, Title } from "@mantine/core";
import { useState, type ReactNode } from "react";
import { useCreateUser, useLogin, useSession, useUsers } from "../api/hooks";

/** Без сессии — выбор пользователя («войти как») или создание нового; пароля нет, приложение локальное. */
export default function UserGate({ children }: { children: ReactNode }) {
  const session = useSession();
  const users = useUsers();
  const login = useLogin();
  const create = useCreateUser();
  const [name, setName] = useState("");

  if (session.isLoading) return <Center h="100vh"><Loader /></Center>;
  if (session.data) return <>{children}</>;

  return (
    <Center h="100vh" px="md">
      <Paper withBorder p="xl" w={360} maw="100%">
        <Stack>
          <Title order={3}>Кто читает?</Title>
          <Select placeholder="Выберите себя"
                  data={(users.data ?? []).map((u) => ({ value: String(u.id), label: u.name }))}
                  onChange={(v) => v && login.mutate(Number(v))} />
          <TextInput label="Или новый пользователь" value={name} onChange={(e) => setName(e.currentTarget.value)}
                     error={create.error?.message} />
          <Button disabled={!name.trim()} loading={create.isPending || login.isPending}
                  onClick={async () => login.mutate((await create.mutateAsync(name.trim())).id)}>
            Создать и войти
          </Button>
        </Stack>
      </Paper>
    </Center>
  );
}
