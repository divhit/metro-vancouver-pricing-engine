"use client";

import { UserButton } from "@clerk/nextjs";

export default function ClerkUserButton() {
  return (
    <UserButton
      afterSignOutUrl="/"
      appearance={{ elements: { avatarBox: "w-8 h-8" } }}
    />
  );
}
