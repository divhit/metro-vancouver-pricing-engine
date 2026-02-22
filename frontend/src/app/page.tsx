import Link from "next/link";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-sand-50 bg-dotgrid relative overflow-hidden">
      {/* Top bar */}
      <header className="relative z-10 flex items-center justify-between px-8 py-5">
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-teal-500 to-teal-700 flex items-center justify-center">
            <span className="text-white font-semibold text-base" style={{ fontFamily: "var(--font-display)" }}>V</span>
          </div>
          <span className="text-2xl tracking-tight text-sand-900" style={{ fontFamily: "var(--font-display)" }}>
            Valuo
          </span>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/sign-in"
            className="px-4 py-2 text-sm font-medium text-sand-600 hover:text-sand-900 transition"
          >
            Sign In
          </Link>
          <Link
            href="/sign-up"
            className="px-5 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-teal-600 to-teal-700 rounded-lg hover:from-teal-700 hover:to-teal-800 transition shadow-sm"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative z-10 flex flex-col items-center text-center px-6 pt-24 pb-32">
        <div className="animate-fade-in-up opacity-0 delay-1">
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-700 text-xs font-medium mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-500 pulse-soft" />
            ML-Powered Valuations for Metro Vancouver
          </span>
        </div>

        <h1
          className="animate-fade-in-up opacity-0 delay-2 text-6xl md:text-7xl lg:text-8xl tracking-tight text-sand-900 max-w-4xl leading-[0.95]"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Property intelligence,{" "}
          <span className="text-gradient italic">refined.</span>
        </h1>

        <p className="animate-fade-in-up opacity-0 delay-3 mt-6 text-lg text-sand-500 max-w-xl leading-relaxed">
          Comprehensive valuations powered by BC Assessment data, sub-region ML
          models, and comparable analysis across 22 Vancouver neighbourhoods.
        </p>

        <div className="animate-fade-in-up opacity-0 delay-4 mt-10 flex items-center gap-4">
          <Link
            href="/dashboard"
            className="px-8 py-3.5 text-sm font-semibold text-white bg-gradient-to-r from-teal-600 to-teal-700 rounded-xl hover:from-teal-700 hover:to-teal-800 transition shadow-md shadow-teal-500/20"
          >
            Start Valuating
          </Link>
          <Link
            href="/sign-in"
            className="px-6 py-3.5 text-sm font-medium text-sand-600 border border-sand-300 rounded-xl hover:bg-white hover:border-sand-400 transition"
          >
            Sign In
          </Link>
        </div>

        {/* Stats row */}
        <div className="animate-fade-in-up opacity-0 delay-5 mt-20 flex items-center gap-12">
          {[
            { value: "22", label: "Neighbourhoods" },
            { value: "200K+", label: "Properties" },
            { value: "150+", label: "Features" },
          ].map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-3xl font-light text-sand-800" style={{ fontFamily: "var(--font-display)" }}>
                {s.value}
              </div>
              <div className="text-xs text-sand-400 mt-1 tracking-wide uppercase">
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Feature cards */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 pb-24">
        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              title: "Sub-Region Models",
              desc: "LightGBM models trained per neighbourhood and property type. No citywide averaging — each micro-market gets its own model.",
              icon: "M3 10l7-7 7 7M6 7v10a1 1 0 001 1h3m4 0h3a1 1 0 001-1V7",
            },
            {
              title: "Explainable AI",
              desc: "SHAP-based feature explanations show exactly what drives each valuation. Transparent, defensible, client-ready.",
              icon: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z",
            },
            {
              title: "Market Intelligence",
              desc: "Real-time market context with YoY trends, interest rate sensitivity, and supply pipeline tracking across Metro Vancouver.",
              icon: "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6",
            },
          ].map((f) => (
            <div key={f.title} className="card-hairline p-6 hover:border-teal-300 transition-colors duration-300">
              <div className="w-10 h-10 rounded-lg bg-teal-50 border border-teal-200 flex items-center justify-center mb-4">
                <svg className="w-5 h-5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d={f.icon} />
                </svg>
              </div>
              <h3 className="text-base font-semibold text-sand-800 mb-2">{f.title}</h3>
              <p className="text-sm text-sand-500 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-sand-200 py-6 text-center">
        <p className="text-xs text-sand-400">
          Valuo &mdash; Metro Vancouver Property Intelligence Engine &middot; Built for professionals
        </p>
      </footer>
    </div>
  );
}
