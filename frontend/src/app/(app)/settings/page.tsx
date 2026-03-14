"use client";

export default function SettingsPage() {
  return (
    <div className="space-y-8">
      <div className="border-b-2 border-concrete-900 pb-4">
        <h1
          className="text-3xl sm:text-4xl text-concrete-900 tracking-[0.08em]"
          style={{ fontFamily: "var(--font-display)" }}
        >
          CONFIG
        </h1>
        <p className="text-concrete-500 text-[11px] mt-1 tracking-wider">
          MANAGE ACCOUNT & PREFERENCES
        </p>
      </div>

      <div className="card-hairline p-6 sm:p-8 space-y-6">
        <div>
          <h2 className="text-[10px] font-bold text-concrete-900 mb-3 tracking-[0.2em]">ACCOUNT</h2>
          <p className="text-[11px] text-concrete-500 tracking-wide leading-relaxed">
            Sign in with Clerk to manage your profile, security settings, and connected accounts.
            Configure your Clerk publishable key in <code className="px-1.5 py-0.5 bg-concrete-100 border border-concrete-200 text-[10px] font-bold">.env.local</code> to enable authentication.
          </p>
        </div>

        <hr className="border-concrete-900 border-t-2" />

        <div>
          <h2 className="text-[10px] font-bold text-concrete-900 mb-3 tracking-[0.2em]">API CONFIGURATION</h2>
          <div className="space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between py-2 gap-2">
              <div>
                <div className="text-[11px] text-concrete-700 font-bold tracking-wider">BACKEND API URL</div>
                <div className="text-[10px] text-concrete-400 mt-0.5 tracking-wider">Where the pricing engine API is hosted</div>
              </div>
              <code className="text-[10px] px-3 py-1.5 bg-concrete-100 border-2 border-concrete-900 text-concrete-600 font-bold break-all">
                {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
              </code>
            </div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between py-2 gap-2">
              <div>
                <div className="text-[11px] text-concrete-700 font-bold tracking-wider">CLERK AUTHENTICATION</div>
                <div className="text-[10px] text-concrete-400 mt-0.5 tracking-wider">User authentication and session management</div>
              </div>
              <span className="text-[10px] px-3 py-1 bg-signal text-white font-bold tracking-[0.15em]">
                NOT CONFIGURED
              </span>
            </div>
          </div>
        </div>

        <hr className="border-concrete-900 border-t-2" />

        <div>
          <h2 className="text-[10px] font-bold text-concrete-900 mb-3 tracking-[0.2em]">SETUP INSTRUCTIONS</h2>
          <ol className="space-y-2 text-[11px] text-concrete-600 tracking-wide">
            <li className="flex gap-3">
              <span className="text-signal font-bold tracking-[0.2em]">01//</span>
              Create a Clerk account at <a href="https://clerk.com" target="_blank" className="text-signal font-bold underline ml-1">clerk.com</a>
            </li>
            <li className="flex gap-3">
              <span className="text-signal font-bold tracking-[0.2em]">02//</span>
              Copy your publishable key and secret key
            </li>
            <li className="flex gap-3">
              <span className="text-signal font-bold tracking-[0.2em]">03//</span>
              Update <code className="px-1 py-0.5 bg-concrete-100 border border-concrete-200 text-[10px] font-bold">.env.local</code> with your keys
            </li>
            <li className="flex gap-3">
              <span className="text-signal font-bold tracking-[0.2em]">04//</span>
              Restart the development server
            </li>
          </ol>
        </div>
      </div>
    </div>
  );
}
