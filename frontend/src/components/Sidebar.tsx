"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import dynamic from "next/dynamic";

const clerkKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
const hasClerk = clerkKey && !clerkKey.startsWith("pk_test_placeholder");

const ClerkUserButton = hasClerk
  ? dynamic(() => import("./ClerkUserButton"), { ssr: false })
  : null;

const NAV_ITEMS = [
  { href: "/dashboard", label: "DASH", num: "01" },
  { href: "/valuation", label: "VALUE", num: "02" },
  { href: "/market", label: "MARKET", num: "03" },
  { href: "/settings", label: "CONFIG", num: "04" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  useEffect(() => { setOpen(false); }, [pathname]);

  useEffect(() => {
    if (open) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  return (
    <>
      {/* Mobile top bar */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-concrete-900 flex items-center justify-between px-4 py-3">
        <Link href="/dashboard" className="flex items-center gap-3">
          <span
            className="text-2xl text-white tracking-widest"
            style={{ fontFamily: "var(--font-display)" }}
          >
            VALUO
          </span>
          <span className="text-[9px] text-signal font-bold tracking-widest">// PROPERTY INTEL</span>
        </Link>
        <button
          onClick={() => setOpen(!open)}
          className="p-2 text-white hover:text-signal transition-colors"
          aria-label="Toggle menu"
        >
          {open ? (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="square">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="square">
              <path d="M4 7h16M4 12h12M4 17h16" />
            </svg>
          )}
        </button>
      </div>

      {/* Mobile overlay */}
      {open && (
        <div
          className="lg:hidden fixed inset-0 bg-black/60 z-40"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 bottom-0 w-[220px] bg-concrete-900 border-r-2 border-signal flex flex-col z-50 transition-transform duration-200 ${
          open ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0`}
      >
        {/* Logo */}
        <div className="px-5 py-6 border-b border-concrete-700">
          <Link href="/dashboard" className="block">
            <div
              className="text-3xl text-white tracking-[0.2em]"
              style={{ fontFamily: "var(--font-display)" }}
            >
              VALUO
            </div>
            <div className="text-[9px] text-signal font-bold tracking-[0.3em] mt-0.5">
              // PROPERTY INTEL
            </div>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-2">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-5 py-3 text-[13px] font-bold tracking-[0.15em] transition-all duration-100 border-l-[3px] ${
                  isActive
                    ? "border-l-signal bg-signal/10 text-signal"
                    : "border-l-transparent text-concrete-400 hover:text-white hover:bg-concrete-800"
                }`}
              >
                <span className="text-[10px] text-concrete-500 w-5 tabular-nums">{item.num}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Version / User */}
        <div className="px-5 py-4 border-t border-concrete-700">
          {ClerkUserButton ? (
            <div className="flex items-center gap-3">
              <ClerkUserButton />
              <span className="text-[10px] text-concrete-500 tracking-wider">ACCOUNT</span>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 bg-signal flex items-center justify-center text-white text-[11px] font-bold">
                A
              </div>
              <div>
                <div className="text-[11px] text-concrete-300 font-bold tracking-wider">APARNA</div>
                <div className="text-[9px] text-concrete-500">v2.0 // prod</div>
              </div>
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
