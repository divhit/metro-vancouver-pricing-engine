import { SignIn } from "@clerk/nextjs";

export default function SignInPage() {
  return (
    <div className="min-h-screen bg-sand-50 bg-dotgrid flex items-center justify-center">
      <SignIn
        appearance={{
          elements: {
            rootBox: "mx-auto",
            card: "shadow-none border border-sand-200 rounded-xl",
          },
        }}
      />
    </div>
  );
}
