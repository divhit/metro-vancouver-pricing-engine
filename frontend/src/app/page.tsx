import Link from "next/link";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-concrete-100 bg-dotgrid relative overflow-hidden">
      {/* Top bar */}
      <header className="relative z-10 flex items-center justify-between px-4 sm:px-8 py-4 sm:py-5 border-b-2 border-concrete-900">
        <div className="flex items-center gap-3">
          <span
            className="text-2xl sm:text-3xl text-concrete-900 tracking-[0.2em]"
            style={{ fontFamily: "var(--font-display)" }}
          >
            VALUO
          </span>
          <span className="hidden sm:inline text-[9px] text-signal font-bold tracking-[0.3em]">// PROPERTY INTEL</span>
        </div>
        <div className="flex items-center gap-2 sm:gap-3">
          <Link
            href="/sign-in"
            className="px-4 py-2 text-[11px] font-bold tracking-wider text-concrete-600 hover:text-concrete-900 transition uppercase"
          >
            Sign In
          </Link>
          <Link
            href="/sign-up"
            className="px-5 py-2.5 text-[11px] font-bold tracking-wider text-white bg-concrete-900 hover:bg-signal transition uppercase"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative z-10 flex flex-col items-center text-center px-4 sm:px-6 pt-16 sm:pt-28 pb-20 sm:pb-32">
        <div className="animate-fade-in-up opacity-0 delay-1">
          <span className="inline-flex items-center gap-2 px-4 py-1.5 bg-signal text-white text-[10px] font-bold tracking-[0.2em] uppercase mb-8">
            <span className="w-2 h-2 bg-white pulse-soft" />
            ML-POWERED // METRO VANCOUVER
          </span>
        </div>

        <h1
          className="animate-fade-in-up opacity-0 delay-2 text-5xl sm:text-7xl md:text-8xl lg:text-[10rem] tracking-[0.05em] text-concrete-900 max-w-5xl leading-[0.85]"
          style={{ fontFamily: "var(--font-display)" }}
        >
          PROPERTY
          <br />
          <span className="text-signal">INTELLIGENCE</span>
        </h1>

        <p className="animate-fade-in-up opacity-0 delay-3 mt-6 sm:mt-8 text-xs sm:text-sm text-concrete-500 max-w-lg leading-relaxed tracking-wide">
          Comprehensive valuations powered by BC Assessment data, sub-region ML
          models, and comparable analysis across 22 Vancouver neighbourhoods.
        </p>

        <div className="animate-fade-in-up opacity-0 delay-4 mt-8 sm:mt-12 flex flex-col sm:flex-row items-center gap-3 sm:gap-4 w-full sm:w-auto">
          <Link
            href="/dashboard"
            className="w-full sm:w-auto text-center px-10 py-4 text-[11px] font-bold tracking-[0.2em] text-white bg-signal hover:bg-signal-dark transition uppercase"
          >
            START VALUATING
          </Link>
          <Link
            href="/sign-in"
            className="w-full sm:w-auto text-center px-8 py-4 text-[11px] font-bold tracking-[0.2em] text-concrete-900 border-2 border-concrete-900 hover:bg-concrete-900 hover:text-white transition uppercase"
          >
            SIGN IN
          </Link>
        </div>

        {/* Stats row */}
        <div className="animate-fade-in-up opacity-0 delay-5 mt-16 sm:mt-24 flex items-center gap-0">
          {[
            { value: "22", label: "Neighbourhoods" },
            { value: "200K+", label: "Properties" },
            { value: "150+", label: "Features" },
          ].map((s, i) => (
            <div key={s.label} className={`text-center px-6 sm:px-10 ${i > 0 ? "border-l-2 border-concrete-900" : ""}`}>
              <div
                className="text-3xl sm:text-5xl text-concrete-900"
                style={{ fontFamily: "var(--font-display)", letterSpacing: "0.05em" }}
              >
                {s.value}
              </div>
              <div className="text-[9px] text-concrete-500 mt-1 tracking-[0.2em] uppercase font-bold">
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Feature cards */}
      <section className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 pb-16 sm:pb-24">
        <div className="grid md:grid-cols-3 gap-0 border-2 border-concrete-900">
          {[
            {
              title: "SUB-REGION MODELS",
              desc: "LightGBM models trained per neighbourhood and property type. No citywide averaging \u2014 each micro-market gets its own model.",
              num: "01",
            },
            {
              title: "EXPLAINABLE AI",
              desc: "SHAP-based feature explanations show exactly what drives each valuation. Transparent, defensible, client-ready.",
              num: "02",
            },
            {
              title: "MARKET INTEL",
              desc: "Real-time market context with YoY trends, interest rate sensitivity, and supply pipeline tracking across Metro Vancouver.",
              num: "03",
            },
          ].map((f, i) => (
            <div
              key={f.title}
              className={`p-6 sm:p-8 hover:bg-signal-bg transition-colors duration-150 ${
                i > 0 ? "border-t-2 md:border-t-0 md:border-l-2 border-concrete-900" : ""
              }`}
            >
              <div className="text-[10px] text-signal font-bold tracking-[0.3em] mb-3">
                {f.num} //
              </div>
              <h3
                className="text-xl sm:text-2xl text-concrete-900 mb-3 tracking-wide"
                style={{ fontFamily: "var(--font-display)" }}
              >
                {f.title}
              </h3>
              <p className="text-[11px] text-concrete-500 leading-relaxed tracking-wide">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t-2 border-concrete-900 py-5 text-center bg-concrete-900">
        <p className="text-[10px] text-concrete-400 tracking-[0.2em] uppercase font-bold">
          VALUO // METRO VANCOUVER PROPERTY INTELLIGENCE ENGINE // BUILT FOR PROFESSIONALS
        </p>
      </footer>
    </div>
  );
}
