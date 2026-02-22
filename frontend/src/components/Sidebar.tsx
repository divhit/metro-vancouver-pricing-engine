"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import dynamic from "next/dynamic";

const clerkKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
const hasClerk = clerkKey && !clerkKey.startsWith("pk_test_placeholder");

const ClerkUserButton = hasClerk
  ? dynamic(() => import("./ClerkUserButton"), { ssr: false })
  : null;

const NAV_ITEMS = [
  {
    href: "/dashboard",
    label: "Dashboard",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="2" width="7" height="8" rx="1.5" />
        <rect x="11" y="2" width="7" height="5" rx="1.5" />
        <rect x="2" y="12" width="7" height="6" rx="1.5" />
        <rect x="11" y="9" width="7" height="9" rx="1.5" />
      </svg>
    ),
  },
  {
    href: "/valuation",
    label: "Valuation",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10 2L3 7v10a1 1 0 001 1h4v-5a2 2 0 014 0v5h4a1 1 0 001-1V7l-7-5z" />
      </svg>
    ),
  },
  {
    href: "/market",
    label: "Market",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="2 16 6 10 10 13 14 6 18 9" />
        <line x1="2" y1="18" x2="18" y2="18" />
      </svg>
    ),
  },
  {
    href: "/settings",
    label: "Settings",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="10" cy="10" r="3" />
        <path d="M10 1.5v2M10 16.5v2M3.7 3.7l1.4 1.4M14.9 14.9l1.4 1.4M1.5 10h2M16.5 10h2M3.7 16.3l1.4-1.4M14.9 5.1l1.4-1.4" />
      </svg>
    ),
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-[220px] bg-white border-r border-sand-200 flex flex-col z-40">
      {/* Logo */}
      <div className="px-6 py-6 border-b border-sand-200">
        <Link href="/dashboard" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-teal-700 flex items-center justify-center">
            <span className="text-white font-semibold text-sm" style={{ fontFamily: "var(--font-display)" }}>V</span>
          </div>
          <span
            className="text-xl tracking-tight text-sand-900"
            style={{ fontFamily: "var(--font-display)" }}
          >
            Valuo
          </span>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all duration-200 ${
                isActive
                  ? "bg-teal-50 text-teal-700 border border-teal-200/60"
                  : "text-sand-500 hover:text-sand-800 hover:bg-sand-100"
              }`}
            >
              <span className={isActive ? "text-teal-600" : "text-sand-400"}>
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* User */}
      <div className="px-4 py-4 border-t border-sand-200 flex items-center gap-3">
        {ClerkUserButton ? (
          <>
            <ClerkUserButton />
            <span className="text-xs text-sand-500 truncate">Account</span>
          </>
        ) : (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center text-teal-700 text-xs font-semibold">
              A
            </div>
            <span className="text-xs text-sand-500">Aparna</span>
          </div>
        )}
      </div>
    </aside>
  );
}
