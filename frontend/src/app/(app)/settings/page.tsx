"use client";

export default function SettingsPage() {
  return (
    <div className="space-y-8">
      <div>
        <h1
          className="text-3xl text-sand-900 tracking-tight"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Settings
        </h1>
        <p className="text-sand-500 text-sm mt-1">
          Manage your account and preferences
        </p>
      </div>

      <div className="card-hairline p-8 space-y-6">
        <div>
          <h2 className="text-sm font-semibold text-sand-800 mb-4">Account</h2>
          <p className="text-sm text-sand-500">
            Sign in with Clerk to manage your profile, security settings, and connected accounts.
            Configure your Clerk publishable key in <code className="px-1.5 py-0.5 bg-sand-100 rounded text-xs">.env.local</code> to enable authentication.
          </p>
        </div>

        <hr className="border-sand-200" />

        <div>
          <h2 className="text-sm font-semibold text-sand-800 mb-4">API Configuration</h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2">
              <div>
                <div className="text-sm text-sand-700">Backend API URL</div>
                <div className="text-xs text-sand-400 mt-0.5">Where the pricing engine API is hosted</div>
              </div>
              <code className="text-xs px-3 py-1.5 bg-sand-100 rounded-lg text-sand-600">
                {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
              </code>
            </div>
            <div className="flex items-center justify-between py-2">
              <div>
                <div className="text-sm text-sand-700">Clerk Authentication</div>
                <div className="text-xs text-sand-400 mt-0.5">User authentication and session management</div>
              </div>
              <span className="text-xs px-2.5 py-1 rounded-full bg-amber-50 text-amber-600 border border-amber-200">
                Not Configured
              </span>
            </div>
          </div>
        </div>

        <hr className="border-sand-200" />

        <div>
          <h2 className="text-sm font-semibold text-sand-800 mb-4">Setup Instructions</h2>
          <ol className="space-y-2 text-sm text-sand-600">
            <li className="flex gap-2">
              <span className="text-teal-600 font-semibold">1.</span>
              Create a Clerk account at <a href="https://clerk.com" target="_blank" className="text-teal-600 underline">clerk.com</a>
            </li>
            <li className="flex gap-2">
              <span className="text-teal-600 font-semibold">2.</span>
              Copy your publishable key and secret key
            </li>
            <li className="flex gap-2">
              <span className="text-teal-600 font-semibold">3.</span>
              Update <code className="px-1 py-0.5 bg-sand-100 rounded text-xs">.env.local</code> with your keys
            </li>
            <li className="flex gap-2">
              <span className="text-teal-600 font-semibold">4.</span>
              Restart the development server
            </li>
          </ol>
        </div>
      </div>
    </div>
  );
}
