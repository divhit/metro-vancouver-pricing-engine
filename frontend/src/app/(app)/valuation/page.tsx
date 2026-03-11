"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  api,
  type PredictionRequest,
  type PredictionResponse,
  type CMAResponse,
  type CMAComparable,
  type SearchResult,
  type BuildingUnit,
} from "@/lib/api";
import {
  formatCurrencyFull,
  formatCurrency,
  formatPercent,
  formatDistance,
  gradeColor,
  gradeBg,
  severityColor,
} from "@/lib/format";

const GOOGLE_MAPS_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_KEY;

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

// Load Google Maps script once
let googleMapsLoaded = false;
function loadGoogleMaps(): Promise<void> {
  if (googleMapsLoaded || typeof window === "undefined") return Promise.resolve();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if ((window as any).google?.maps?.places) {
    googleMapsLoaded = true;
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const existing = document.querySelector('script[src*="maps.googleapis.com"]');
    if (existing) {
      existing.addEventListener("load", () => { googleMapsLoaded = true; resolve(); });
      return;
    }
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_MAPS_KEY}&libraries=places`;
    script.async = true;
    script.onload = () => { googleMapsLoaded = true; resolve(); };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

export default function ValuationPage() {
  const [mode, setMode] = useState<"pid" | "address" | "coordinates">("address");
  const [pid, setPid] = useState("");
  const [address, setAddress] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [propertyType, setPropertyType] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [cmaResult, setCmaResult] = useState<CMAResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // PID autocomplete state
  const [pidQuery, setPidQuery] = useState("");
  const [pidSuggestions, setPidSuggestions] = useState<SearchResult[]>([]);
  const [showPidDropdown, setShowPidDropdown] = useState(false);
  const [pidHighlight, setPidHighlight] = useState(-1);
  const pidDropdownRef = useRef<HTMLDivElement>(null);

  // Google Places autocomplete state
  const addressInputRef = useRef<HTMLInputElement>(null);
  const autocompleteRef = useRef<google.maps.places.Autocomplete | null>(null);
  const [selectedCoords, setSelectedCoords] = useState<{ lat: number; lng: number } | null>(null);
  const [selectedAddress, setSelectedAddress] = useState("");

  // Address disambiguation: when multiple properties match
  const [addressMatches, setAddressMatches] = useState<SearchResult[]>([]);
  const [selectedMatch, setSelectedMatch] = useState<SearchResult | null>(null);

  // Strata building: prompt for unit number
  const [isStrataBuilding, setIsStrataBuilding] = useState(false);
  const [buildingUnits, setBuildingUnits] = useState<BuildingUnit[]>([]);
  const [unitInput, setUnitInput] = useState("");
  const [unitSuggestions, setUnitSuggestions] = useState<BuildingUnit[]>([]);
  const [showUnitDropdown, setShowUnitDropdown] = useState(false);

  const debouncedPidQuery = useDebounce(pidQuery, 250);

  const runPrediction = useCallback(async (overrideReq?: Partial<PredictionRequest>) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setCmaResult(null);

    const req: PredictionRequest = {};
    if (overrideReq) {
      Object.assign(req, overrideReq);
    } else if (mode === "pid" && pid) {
      req.pid = pid;
    } else if (mode === "address") {
      // If user selected a specific property from disambiguation, use its PID
      if (selectedMatch) {
        req.pid = selectedMatch.pid;
      } else if (pid) {
        // PID was set by single-match auto-select
        req.pid = pid;
      } else {
        if (selectedCoords) {
          req.latitude = selectedCoords.lat;
          req.longitude = selectedCoords.lng;
        }
        if (address) req.address = address;
      }
    } else if (mode === "coordinates" && lat && lon) {
      req.latitude = parseFloat(lat);
      req.longitude = parseFloat(lon);
    }
    if (propertyType) req.property_type = propertyType;

    try {
      // Run both prediction and CMA in parallel
      const predictionPromise = api.predict(req);
      const cmaReq = {
        pid: req.pid,
        address: req.address,
        latitude: req.latitude,
        longitude: req.longitude,
        property_type: req.property_type,
        max_comps: 10,
        max_radius_m: 3000,
        max_age_days: 90,
      };
      const cmaPromise = api.getCMA(cmaReq).catch(() => null);

      const [predRes, cmaRes] = await Promise.all([predictionPromise, cmaPromise]);
      setResult(predRes);
      if (cmaRes) setCmaResult(cmaRes);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [mode, pid, address, selectedCoords, selectedMatch, lat, lon, propertyType]);

  // PID mode: fetch from internal DB
  useEffect(() => {
    if (mode !== "pid" || debouncedPidQuery.length < 2) {
      setPidSuggestions([]);
      return;
    }
    let cancelled = false;
    api.searchProperties(debouncedPidQuery, 8).then((results) => {
      if (!cancelled) {
        setPidSuggestions(results);
        setShowPidDropdown(results.length > 0);
        setPidHighlight(-1);
      }
    }).catch(() => {
      if (!cancelled) setPidSuggestions([]);
    });
    return () => { cancelled = true; };
  }, [debouncedPidQuery, mode]);

  // Address mode: init Google Places autocomplete
  useEffect(() => {
    if (mode !== "address" || !GOOGLE_MAPS_KEY) return;

    let mounted = true;
    loadGoogleMaps().then(() => {
      if (!mounted || !addressInputRef.current) return;
      if (autocompleteRef.current) return;

      const ac = new google.maps.places.Autocomplete(addressInputRef.current, {
        componentRestrictions: { country: "ca" },
        fields: ["formatted_address", "geometry", "address_components"],
        types: ["address"],
      });

      const vancouverBounds = new google.maps.LatLngBounds(
        new google.maps.LatLng(49.19, -123.27),
        new google.maps.LatLng(49.32, -123.02),
      );
      ac.setBounds(vancouverBounds);

      ac.addListener("place_changed", () => {
        const place = ac.getPlace();
        if (!place.geometry?.location) return;

        const formattedAddr = place.formatted_address || "";
        const coords = {
          lat: place.geometry.location.lat(),
          lng: place.geometry.location.lng(),
        };

        setAddress(formattedAddr);
        setSelectedAddress(formattedAddr);
        setSelectedCoords(coords);
        setLat(coords.lat.toString());
        setLon(coords.lng.toString());
        setSelectedMatch(null);
        setAddressMatches([]);
        setIsStrataBuilding(false);
        setBuildingUnits([]);
        setUnitInput("");
        setUnitSuggestions([]);

        // Extract street number + full street from Google address and search our DB
        // e.g. "312 East 40th Avenue, Vancouver, BC, Canada" → search "312 40TH AVE E"
        const beforeCity = formattedAddr.replace(/,.*$/, "").trim();

        // Extract unit number from Google address (#302, Suite 302, Unit 302)
        let googleUnit: string | null = null;
        const unitMatch = beforeCity.match(/#\s*(\d+)/);
        if (unitMatch) {
          googleUnit = unitMatch[1];
        } else {
          const suiteMatch = beforeCity.match(/\b(?:Suite|Unit|Apt|Ste)\s*(\d+)/i);
          if (suiteMatch) googleUnit = suiteMatch[1];
        }

        // Remove unit part from address for street matching
        let cleanAddr = beforeCity
          .replace(/#\s*\d+/, "")
          .replace(/\b(?:Suite|Unit|Apt|Ste)\s*\d+/i, "")
          .replace(/,\s*$/, "")
          .trim();

        const streetNum = cleanAddr.match(/^(\d+)/)?.[1];
        if (streetNum) {
          // Normalize: "East 40th Avenue" → "40TH AVE E"
          let streetPart = cleanAddr.replace(/^\d+\s*/, "").trim();
          let dir = "";
          const dirMatch = streetPart.match(/^(East|West|North|South)\b\s*/i);
          if (dirMatch) {
            dir = dirMatch[1].charAt(0).toUpperCase();
            streetPart = streetPart.replace(dirMatch[0], "").trim();
          }
          streetPart = streetPart
            .replace(/\bAvenue\b/i, "AVE")
            .replace(/\bStreet\b/i, "ST")
            .replace(/\bDrive\b/i, "DR")
            .replace(/\bRoad\b/i, "RD")
            .replace(/\bBoulevard\b/i, "BLVD")
            .replace(/\bCrescent\b/i, "CRES")
            .replace(/\bPlace\b/i, "PL")
            .replace(/\bCourt\b/i, "CT");
          const normalizedStreet = `${streetPart}${dir ? " " + dir : ""}`.toUpperCase();

          // Check if this is a strata building with multiple units
          api.getBuildingUnits(parseInt(streetNum), normalizedStreet).then((units) => {
            // Only treat as strata if units are condos/townhomes, NOT duplexes
            const isStrata = units.length > 2 ||
              (units.length > 1 && units.some((u) => u.property_type === "condo" || u.property_type === "townhome"));
            if (isStrata) {
              // This is a strata building (condo/townhome)
              setIsStrataBuilding(true);
              setBuildingUnits(units);
              setPropertyType("condo");

              if (googleUnit) {
                // Google address had a unit number — auto-select it
                const match = units.find((u) => u.unit_number === parseInt(googleUnit!));
                if (match) {
                  setSelectedMatch({
                    pid: match.pid,
                    address: `${googleUnit} ${normalizedStreet}`,
                    property_type: match.property_type,
                    neighbourhood: "",
                    assessed_value: match.assessed_value,
                  });
                  setPid(match.pid);
                  setUnitInput(googleUnit);
                } else {
                  // Unit not found in our data — show input
                  setUnitInput(googleUnit);
                }
              }
              // If no googleUnit, the UI will prompt the user
            } else if (units.length >= 1) {
              // Not strata: either single property or duplex (2 PIDs, same lot)
              // For duplexes: pick the newer PID (higher number = actual unit, not old lot)
              const sorted = [...units].sort((a, b) =>
                parseInt(b.pid) - parseInt(a.pid)
              );
              const best = sorted[0];
              setSelectedMatch({
                pid: best.pid,
                address: `${streetNum} ${normalizedStreet}`,
                property_type: best.property_type,
                neighbourhood: "",
                assessed_value: best.assessed_value,
              });
              setPid(best.pid);
              if (best.property_type) setPropertyType(best.property_type);
            } else {
              // Not a strata building — do normal search
              const searchQuery = `${streetNum} ${normalizedStreet}`;
              api.searchProperties(searchQuery, 10).then((matches) => {
                const exact = matches.filter((m) => {
                  const mAddr = m.address.toUpperCase();
                  return mAddr.includes(streetPart.split(/\s+/)[0].toUpperCase());
                });
                let filtered = exact;
                if (dir && exact.length > 1) {
                  const withDir = exact.filter((m) => m.address.toUpperCase().endsWith(" " + dir));
                  if (withDir.length > 0) filtered = withDir;
                }

                if (filtered.length > 1) {
                  setAddressMatches(filtered);
                } else if (filtered.length === 1) {
                  setSelectedMatch(filtered[0]);
                  setPid(filtered[0].pid);
                  if (filtered[0].property_type) setPropertyType(filtered[0].property_type);
                  setAddressMatches([]);
                } else if (exact.length > 0) {
                  if (exact.length > 1) {
                    setAddressMatches(exact);
                  } else {
                    setSelectedMatch(exact[0]);
                    setPid(exact[0].pid);
                    if (exact[0].property_type) setPropertyType(exact[0].property_type);
                    setAddressMatches([]);
                  }
                }
              }).catch(() => {});
            }
          }).catch(() => {
            // Fallback: normal search if building-units endpoint fails
            const searchQuery = `${streetNum} ${normalizedStreet}`;
            api.searchProperties(searchQuery, 10).then((matches) => {
              const exact = matches.filter((m) => {
                const mAddr = m.address.toUpperCase();
                return mAddr.includes(streetPart.split(/\s+/)[0].toUpperCase());
              });
              if (exact.length > 1) {
                setAddressMatches(exact);
              } else if (exact.length === 1) {
                setSelectedMatch(exact[0]);
                setPid(exact[0].pid);
                if (exact[0].property_type) setPropertyType(exact[0].property_type);
              }
            }).catch(() => {});
          });
        }
      });

      autocompleteRef.current = ac;
    });

    return () => { mounted = false; };
  }, [mode]);

  useEffect(() => {
    if (mode !== "address") {
      autocompleteRef.current = null;
    }
  }, [mode]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (pidDropdownRef.current && !pidDropdownRef.current.contains(e.target as Node)) {
        setShowPidDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectPidSuggestion = useCallback((s: SearchResult) => {
    setPid(s.pid);
    setPidQuery(s.address + " (PID: " + s.pid + ")");
    if (s.property_type) setPropertyType(s.property_type);
    setShowPidDropdown(false);
  }, []);

  function handlePidKeyDown(e: React.KeyboardEvent) {
    if (!showPidDropdown || pidSuggestions.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setPidHighlight((i) => Math.min(i + 1, pidSuggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setPidHighlight((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && pidHighlight >= 0) {
      e.preventDefault();
      selectPidSuggestion(pidSuggestions[pidHighlight]);
    } else if (e.key === "Escape") {
      setShowPidDropdown(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setShowPidDropdown(false);
    // Block submission if disambiguation or unit selection is needed
    if (mode === "address" && addressMatches.length > 1 && !selectedMatch) {
      setError("Please select one of the matching properties above before running valuation.");
      return;
    }
    if (mode === "address" && isStrataBuilding && !selectedMatch) {
      setError("This is a strata building — please enter a unit number above.");
      return;
    }
    runPrediction();
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl text-sand-900 tracking-tight" style={{ fontFamily: "var(--font-display)" }}>
          Property Valuation
        </h1>
        <p className="text-sand-500 text-sm mt-1">
          ML-powered valuation with Comparative Market Analysis
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSubmit} className="card-hairline p-6 space-y-5">
        <div className="flex gap-1 p-1 bg-sand-100 rounded-lg w-fit">
          {(["pid", "address", "coordinates"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`px-4 py-1.5 text-xs font-medium rounded-md transition ${
                mode === m ? "bg-white text-sand-900 shadow-sm" : "text-sand-500 hover:text-sand-700"
              }`}
            >
              {m === "pid" ? "PID" : m === "address" ? "Address" : "Coordinates"}
            </button>
          ))}
        </div>

        <div className="flex gap-4 items-end">
          <div className="flex-1">
            {mode === "pid" && (
              <div className="relative" ref={pidDropdownRef}>
                <label className="block text-xs font-medium text-sand-500 mb-1.5">Search by PID or Street Name</label>
                <div className="relative">
                  <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-sand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input
                    type="text"
                    value={pidQuery}
                    onChange={(e) => { setPidQuery(e.target.value); setPid(e.target.value); }}
                    onFocus={() => pidSuggestions.length > 0 && setShowPidDropdown(true)}
                    onKeyDown={handlePidKeyDown}
                    placeholder="e.g. 012-345-678 or Main St..."
                    autoComplete="off"
                    className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                  />
                </div>
                {showPidDropdown && pidSuggestions.length > 0 && (
                  <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-white border border-sand-200 rounded-xl shadow-lg overflow-hidden max-h-[400px] overflow-y-auto">
                    {pidSuggestions.map((s, i) => (
                      <button
                        key={s.pid}
                        type="button"
                        onClick={() => selectPidSuggestion(s)}
                        className={`w-full text-left px-4 py-3 flex items-center justify-between gap-3 transition ${
                          i === pidHighlight ? "bg-teal-50" : "hover:bg-sand-50"
                        } ${i > 0 ? "border-t border-sand-100" : ""}`}
                      >
                        <div className="min-w-0 flex-1">
                          <div className="text-sm text-sand-900 font-medium truncate">{s.address}</div>
                          <div className="text-xs text-sand-400 mt-0.5 flex items-center gap-2">
                            <span className="font-mono">{s.pid}</span>
                            <span>&middot;</span>
                            <span className="capitalize">{s.property_type}</span>
                            <span>&middot;</span>
                            <span>{s.neighbourhood}</span>
                          </div>
                        </div>
                        <div className="text-right shrink-0">
                          <div className="text-sm font-medium text-sand-700">{formatCurrency(s.assessed_value)}</div>
                          <div className="text-[10px] text-sand-400">assessed</div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {mode === "address" && (
              <div>
                <label className="block text-xs font-medium text-sand-500 mb-1.5">
                  Street Address
                  <span className="ml-2 text-[10px] text-teal-500 font-normal">Powered by Google</span>
                </label>
                <div className="relative">
                  <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-sand-400 z-10 pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <input
                    ref={addressInputRef}
                    type="text"
                    placeholder="Start typing an address... e.g. 6149 Fremlin Street"
                    autoComplete="off"
                    className="google-pac-input w-full pl-10 pr-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                  />
                </div>
                {selectedCoords && (
                  <div className="mt-2 space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-teal-50 border border-teal-200 text-[11px] text-teal-700">
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                        Address found
                      </span>
                      <span className="text-[10px] text-sand-400">{selectedCoords.lat.toFixed(5)}, {selectedCoords.lng.toFixed(5)}</span>
                    </div>

                    {/* Strata building: prompt for unit number */}
                    {isStrataBuilding && !selectedMatch && (
                      <div className="p-3 rounded-lg border border-teal-200 bg-teal-50/50">
                        <p className="text-xs font-medium text-teal-800 mb-2">
                          This is a strata building with {buildingUnits.length} units — enter your unit number:
                        </p>
                        <div className="relative">
                          <input
                            type="text"
                            value={unitInput}
                            onChange={(e) => {
                              const val = e.target.value;
                              setUnitInput(val);
                              if (val.length > 0) {
                                const num = parseInt(val);
                                const filtered = buildingUnits.filter((u) =>
                                  u.unit_number !== null && u.unit_number.toString().startsWith(val)
                                );
                                setUnitSuggestions(filtered.slice(0, 10));
                                setShowUnitDropdown(filtered.length > 0);
                                // Auto-select if exact match
                                const exact = buildingUnits.find((u) => u.unit_number === num);
                                if (exact) {
                                  setSelectedMatch({
                                    pid: exact.pid,
                                    address: `Unit ${exact.unit_number}`,
                                    property_type: exact.property_type,
                                    neighbourhood: "",
                                    assessed_value: exact.assessed_value,
                                  });
                                  setPid(exact.pid);
                                  setShowUnitDropdown(false);
                                }
                              } else {
                                setUnitSuggestions([]);
                                setShowUnitDropdown(false);
                              }
                            }}
                            placeholder="e.g. 302, 1201..."
                            autoComplete="off"
                            className="w-full px-3 py-2 rounded-md border border-teal-300 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                          />
                          {showUnitDropdown && unitSuggestions.length > 0 && (
                            <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-white border border-sand-200 rounded-lg shadow-lg overflow-hidden max-h-[200px] overflow-y-auto">
                              {unitSuggestions.map((u) => (
                                <button
                                  key={u.pid}
                                  type="button"
                                  onClick={() => {
                                    setSelectedMatch({
                                      pid: u.pid,
                                      address: `Unit ${u.unit_number}`,
                                      property_type: u.property_type,
                                      neighbourhood: "",
                                      assessed_value: u.assessed_value,
                                    });
                                    setPid(u.pid);
                                    setUnitInput(String(u.unit_number));
                                    setShowUnitDropdown(false);
                                  }}
                                  className="w-full text-left px-3 py-2 text-xs hover:bg-teal-50 transition flex items-center justify-between border-t border-sand-100 first:border-t-0"
                                >
                                  <span className="font-medium">Unit {u.unit_number}</span>
                                  <span className="text-sand-500">{formatCurrency(u.assessed_value)}</span>
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Disambiguation: multiple properties at this address */}
                    {addressMatches.length > 1 && (
                      <div className="p-3 rounded-lg border border-amber-200 bg-amber-50/50">
                        <p className="text-xs font-medium text-amber-800 mb-2">
                          Multiple properties found at this address — please select one:
                        </p>
                        <div className="space-y-1.5">
                          {addressMatches.map((m) => (
                            <button
                              key={m.pid}
                              type="button"
                              onClick={() => {
                                setSelectedMatch(m);
                                setPid(m.pid);
                                if (m.property_type) setPropertyType(m.property_type);
                                setAddressMatches([]);
                              }}
                              className={`w-full text-left px-3 py-2 rounded-md border text-xs transition ${
                                selectedMatch?.pid === m.pid
                                  ? "border-teal-400 bg-teal-50 text-teal-800"
                                  : "border-sand-200 bg-white hover:border-teal-300 hover:bg-teal-50/30 text-sand-700"
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div>
                                  <span className="font-medium">{m.address}</span>
                                  <span className="ml-2 text-sand-400">PID: {m.pid}</span>
                                </div>
                                <div className="flex items-center gap-3">
                                  <span className="px-1.5 py-0.5 rounded bg-sand-100 text-[10px] font-medium uppercase">
                                    {m.property_type}
                                  </span>
                                  <span className="text-sand-500">
                                    {m.assessed_value ? `$${(m.assessed_value / 1000).toFixed(0)}K` : ""}
                                  </span>
                                </div>
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Show selected property confirmation */}
                    {selectedMatch && addressMatches.length === 0 && (
                      <div className="flex items-center gap-2 text-[11px] text-teal-700">
                        <span className="font-medium">
                          {isStrataBuilding ? `${selectedAddress?.replace(/,.*$/, "")} — Unit ${unitInput}` : selectedMatch.address}
                        </span>
                        <span className="px-1.5 py-0.5 rounded bg-teal-50 border border-teal-200 text-[10px] font-medium uppercase">
                          {selectedMatch.property_type}
                        </span>
                        {selectedMatch.assessed_value > 0 && (
                          <span className="text-sand-500">{formatCurrency(selectedMatch.assessed_value)}</span>
                        )}
                        <span className="text-sand-400">PID: {selectedMatch.pid}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            {mode === "coordinates" && (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-sand-500 mb-1.5">Latitude</label>
                  <input type="text" value={lat} onChange={(e) => setLat(e.target.value)} placeholder="49.2827"
                    className="w-full px-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition" />
                </div>
                <div>
                  <label className="block text-xs font-medium text-sand-500 mb-1.5">Longitude</label>
                  <input type="text" value={lon} onChange={(e) => setLon(e.target.value)} placeholder="-123.1207"
                    className="w-full px-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition" />
                </div>
              </div>
            )}
          </div>

          <div className="w-[200px] shrink-0">
            <label className="block text-xs font-medium text-sand-500 mb-1.5">Property Type</label>
            <select value={propertyType} onChange={(e) => setPropertyType(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition">
              <option value="">Auto-detect</option>
              <option value="condo">Condo</option>
              <option value="townhome">Townhome</option>
              <option value="detached">Detached</option>
              <option value="development_land">Development Land</option>
            </select>
          </div>

          <button type="submit" disabled={loading}
            className="shrink-0 px-6 py-2.5 text-sm font-semibold text-white bg-gradient-to-r from-teal-600 to-teal-700 rounded-lg hover:from-teal-700 hover:to-teal-800 transition shadow-sm disabled:opacity-50 disabled:cursor-not-allowed">
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Valuating...
              </span>
            ) : "Get Valuation"}
          </button>
        </div>
      </form>

      {error && (
        <div className="card-hairline p-4 border-rose-200 bg-rose-50">
          <p className="text-sm text-rose-600">{error}</p>
        </div>
      )}

      {result && <ValuationResult result={result} cma={cmaResult} />}
    </div>
  );
}

/* ================================================================
   VALUATION RESULT — Full report with assessment, CMA, breakdown
   ================================================================ */

function ValuationResult({ result, cma }: { result: PredictionResponse; cma: CMAResponse | null }) {
  const [activeTab, setActiveTab] = useState<"overview" | "cma" | "details">("overview");

  const assessedValue = result.assessed_value; // actual BC Assessment value for this property
  const marketEstimate = result.market_estimate || result.point_estimate; // SAR-adjusted market value
  const marketInfo = result.market_model_info;
  const primaryEstimate = result.point_estimate; // best estimate (market estimate if available)

  // Extract SAR and MAPE from market model info
  let sarValue: number | null = null;
  let mapeValue: number | null = null;
  if (marketInfo) {
    const sarMatch = marketInfo.match(/SAR=([\d.]+)/);
    const mapeMatch = marketInfo.match(/MAPE=([\d.]+)%/);
    if (sarMatch) sarValue = parseFloat(sarMatch[1]);
    if (mapeMatch) mapeValue = parseFloat(mapeMatch[1]);
  }

  return (
    <div className="space-y-6 animate-fade-in-up" style={{ animationFillMode: "forwards" }}>

      {/* ===== HERO: Primary Estimate ===== */}
      <div className="card-hairline p-8">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs font-medium text-sand-400 uppercase tracking-wider mb-2">
              Estimated Market Value
            </div>
            <div className="hero-number">{formatCurrencyFull(primaryEstimate)}</div>
            <div className="mt-3 flex items-center gap-4">
              <span className="text-sm text-sand-500">
                {formatCurrencyFull(result.confidence_interval.lower)} &ndash; {formatCurrencyFull(result.confidence_interval.upper)}
              </span>
              <span className="text-xs text-sand-400">
                {(result.confidence_interval.level * 100).toFixed(0)}% confidence interval
              </span>
            </div>
            {/* Quick summary of what drove this */}
            {marketEstimate && (
              <div className="mt-4 text-xs text-sand-500 leading-relaxed max-w-xl">
                Based on BC Assessment value adjusted by Sale-to-Assessment Ratio (SAR) from{" "}
                {marketInfo ? marketInfo.replace(/market_/, "").replace(/\(/, " (") : "recent sold data"}.
                {cma && cma.comparable_count > 0 && (
                  <> Validated against {cma.comparable_count} comparable recent sale{cma.comparable_count > 1 ? "s" : ""}.</>
                )}
              </div>
            )}
          </div>
          <div className="text-right">
            <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border ${gradeBg(result.confidence_grade)}`}>
              <span className={`text-2xl font-semibold ${gradeColor(result.confidence_grade)}`} style={{ fontFamily: "var(--font-display)" }}>
                {result.confidence_grade}
              </span>
              <span className="text-xs text-sand-500">Confidence<br />Grade</span>
            </div>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="mt-6 relative">
          <div className="h-2 bg-sand-100 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-teal-400 to-teal-600 transition-all duration-1000"
              style={{
                width: `${Math.min(100, ((primaryEstimate - result.confidence_interval.lower) / (result.confidence_interval.upper - result.confidence_interval.lower)) * 100)}%`,
              }}
            />
          </div>
          <div className="flex justify-between mt-1.5 text-[10px] text-sand-400">
            <span>{formatCurrency(result.confidence_interval.lower)}</span>
            <span>{formatCurrency(result.confidence_interval.upper)}</span>
          </div>
        </div>
      </div>

      {/* ===== TAB NAVIGATION ===== */}
      <div className="flex gap-1 p-1 bg-sand-100 rounded-lg w-fit">
        {[
          { key: "overview" as const, label: "Valuation Breakdown" },
          { key: "cma" as const, label: `CMA Report${cma ? ` (${cma.comparable_count})` : ""}` },
          { key: "details" as const, label: "Details & Risk" },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-5 py-2 text-xs font-medium rounded-md transition ${
              activeTab === tab.key ? "bg-white text-sand-900 shadow-sm" : "text-sand-500 hover:text-sand-700"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ===== TAB: VALUATION BREAKDOWN ===== */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* Three valuation methods side by side */}
          <div className="grid md:grid-cols-3 gap-4">
            {/* Assessment-Based */}
            <div className="card-hairline p-5">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 rounded-full bg-sky-500" />
                <h4 className="text-xs font-semibold text-sand-700 uppercase tracking-wider">BC Assessment</h4>
              </div>
              <div className="text-2xl font-semibold text-sand-900" style={{ fontFamily: "var(--font-display)" }}>
                {assessedValue ? formatCurrencyFull(assessedValue) : "N/A"}
              </div>
              <p className="text-xs text-sand-400 mt-2">
                Government-assessed value from BC Assessment Authority ({result.market_context.assessment_year}).
                This is the starting point — not necessarily market value.
              </p>
            </div>

            {/* SAR Market Estimate */}
            <div className="card-hairline p-5 border-teal-200 bg-teal-50/30">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 rounded-full bg-teal-500" />
                <h4 className="text-xs font-semibold text-sand-700 uppercase tracking-wider">Market Value (SAR)</h4>
              </div>
              <div className="text-2xl font-semibold text-sand-900" style={{ fontFamily: "var(--font-display)" }}>
                {formatCurrencyFull(marketEstimate)}
              </div>
              <p className="text-xs text-sand-400 mt-2">
                {assessedValue ? (
                  <>
                    BC Assessment ({formatCurrency(assessedValue)}) adjusted by{" "}
                    {sarValue ? <span className="font-medium text-sand-600">{sarValue.toFixed(3)}x SAR</span> : "sale-to-assessment ratio"}{" "}
                    based on how properties in this sub-region actually sell vs. their assessed values.
                    {mapeValue && <> Accuracy: {mapeValue.toFixed(1)}% MAPE.</>}
                  </>
                ) : (
                  <>
                    Derived from how properties in this sub-region sell relative to their assessed values.
                    {sarValue && <> SAR: {sarValue.toFixed(3)}x.</>}
                  </>
                )}
              </p>
            </div>

            {/* CMA Estimate */}
            <div className={`card-hairline p-5 ${cma?.cma_estimate ? "border-emerald-200 bg-emerald-50/30" : ""}`}>
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <h4 className="text-xs font-semibold text-sand-700 uppercase tracking-wider">CMA Estimate</h4>
              </div>
              <div className="text-2xl font-semibold text-sand-900" style={{ fontFamily: "var(--font-display)" }}>
                {cma?.cma_estimate ? formatCurrencyFull(cma.cma_estimate) : "N/A"}
              </div>
              <p className="text-xs text-sand-400 mt-2">
                {cma?.cma_estimate ? (
                  <>
                    Median of {cma.comparable_count} adjusted comparable sales
                    within {formatDistance(cma.market_stats?.avg_distance_m || 0)} radius.
                    {cma.cma_range && (
                      <> Range: {formatCurrency(cma.cma_range.low)} &ndash; {formatCurrency(cma.cma_range.high)}.</>
                    )}
                  </>
                ) : (
                  "No comparable recent sales found within search area."
                )}
              </p>
            </div>
          </div>

          {/* How We Arrived at the Estimate */}
          <div className="card-hairline p-6">
            <h3 className="text-sm font-semibold text-sand-800 mb-5">How We Arrived at This Estimate</h3>
            <div className="space-y-4">
              {/* Step 1 */}
              <div className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-7 h-7 rounded-full bg-sky-100 text-sky-700 flex items-center justify-center text-xs font-bold">1</div>
                  <div className="w-px flex-1 bg-sand-200 mt-1" />
                </div>
                <div className="pb-4">
                  <div className="text-sm font-medium text-sand-800">BC Assessment Base Value</div>
                  <p className="text-xs text-sand-500 mt-1">
                    Started with the BC Assessment Authority&apos;s assessed value for this property,
                    which reflects land value + improvement value as of {result.market_context.assessment_year}.
                    This is not necessarily what the property would sell for — it&apos;s a government estimate used for taxation.
                  </p>
                  <div className="mt-2 text-sm font-medium text-sand-700">
                    {assessedValue ? formatCurrencyFull(assessedValue) : "N/A"}
                  </div>
                </div>
              </div>

              {/* Step 2 */}
              <div className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-7 h-7 rounded-full bg-teal-100 text-teal-700 flex items-center justify-center text-xs font-bold">2</div>
                  <div className="w-px flex-1 bg-sand-200 mt-1" />
                </div>
                <div className="pb-4">
                  <div className="text-sm font-medium text-sand-800">Sub-Region Market Performance (SAR)</div>
                  <p className="text-xs text-sand-500 mt-1">
                    Analyzed how properties in the <span className="font-medium text-sand-700">{result.market_context.neighbourhood_name}</span> sub-region
                    actually sell compared to their assessed values.
                    {sarValue && (
                      <> Properties here have been selling at <span className="font-medium text-sand-700">{(sarValue * 100).toFixed(1)}%</span> of their assessed values.</>
                    )}{" "}
                    Multiplying the assessed value by this ratio gives us the market-derived estimate.
                  </p>
                  <div className="mt-2 flex items-center gap-3">
                    <span className="text-sm font-medium text-sand-700">{formatCurrencyFull(marketEstimate)}</span>
                    {sarValue && assessedValue && (
                      <span className={`text-xs px-2 py-0.5 rounded-full ${sarValue >= 1 ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-600"}`}>
                        {assessedValue ? formatCurrency(assessedValue) : ""} &times; {sarValue.toFixed(3)} SAR
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Step 3: Adjustments */}
              {result.adjustments.length > 0 && (
                <div className="flex gap-4">
                  <div className="flex flex-col items-center">
                    <div className="w-7 h-7 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center text-xs font-bold">{marketEstimate ? "3" : "2"}</div>
                    <div className="w-px flex-1 bg-sand-200 mt-1" />
                  </div>
                  <div className="pb-4">
                    <div className="text-sm font-medium text-sand-800">Property-Specific Adjustments</div>
                    <p className="text-xs text-sand-500 mt-1 mb-3">
                      Applied adjustments for property-specific factors that affect value beyond what the base model captures.
                    </p>
                    <div className="space-y-2">
                      {result.adjustments.map((adj, i) => (
                        <div key={i} className="flex items-center justify-between py-1.5 px-3 rounded-lg bg-sand-50">
                          <div>
                            <span className="text-sm text-sand-800">{adj.name}</span>
                            <span className="text-xs text-sand-400 ml-2">{adj.explanation}</span>
                          </div>
                          <div className={`text-sm font-medium ${adj.percentage >= 0 ? "text-emerald-600" : "text-rose-500"}`}>
                            {adj.percentage >= 0 ? "+" : ""}{adj.percentage.toFixed(1)}%
                            <span className="text-xs text-sand-400 ml-1">
                              ({adj.dollar_amount >= 0 ? "+" : ""}{formatCurrency(adj.dollar_amount)})
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Step 4: CMA Validation */}
              {cma && cma.comparable_count > 0 && (
                <div className="flex gap-4">
                  <div className="flex flex-col items-center">
                    <div className="w-7 h-7 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center text-xs font-bold">
                      {result.adjustments.length > 0 ? (marketEstimate ? "4" : "3") : (marketEstimate ? "3" : "2")}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-sand-800">CMA Validation</div>
                    <p className="text-xs text-sand-500 mt-1">
                      Cross-checked against {cma.comparable_count} comparable properties that sold recently
                      within {formatDistance(cma.market_stats?.avg_distance_m || 0)}.
                      {cma.market_stats?.avg_sar && (
                        <> These comps sold at an average of {(cma.market_stats.avg_sar * 100).toFixed(1)}% of their assessed values.</>
                      )}
                      {cma.market_stats?.avg_dom != null && (
                        <> Average {cma.market_stats.avg_dom.toFixed(0)} days on market.</>
                      )}
                    </p>
                    {cma.recommendation?.estimated_range && (
                      <div className="mt-2 text-xs text-sand-500">
                        Blended recommendation: <span className="font-medium text-sand-700">{formatCurrencyFull(cma.recommendation.estimated_range.low)}</span>
                        {" "}&ndash;{" "}
                        <span className="font-medium text-sand-700">{formatCurrencyFull(cma.recommendation.estimated_range.high)}</span>
                        <span className="ml-2 text-sand-400">({cma.recommendation.method})</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Market Context */}
          <div className="card-hairline p-6">
            <h3 className="text-sm font-semibold text-sand-800 mb-4">Neighbourhood Market Context</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
              {[
                { label: "Neighbourhood", value: result.market_context.neighbourhood_name },
                { label: "Median Assessed", value: formatCurrency(result.market_context.median_assessed_value) },
                { label: "YoY Change", value: formatPercent(result.market_context.yoy_change_pct) },
                { label: "Properties", value: result.market_context.property_count.toLocaleString() },
                { label: "Assessment Year", value: result.market_context.assessment_year.toString() },
              ].map((item) => (
                <div key={item.label}>
                  <div className="text-[11px] text-sand-400 uppercase tracking-wider">{item.label}</div>
                  <div className="text-sm font-medium text-sand-800 mt-1">{item.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ===== TAB: CMA REPORT ===== */}
      {activeTab === "cma" && (
        <CMAReportSection cma={cma} />
      )}

      {/* ===== TAB: DETAILS & RISK ===== */}
      {activeTab === "details" && (
        <div className="space-y-6">
          {/* Assessment Comparables */}
          <div className="card-hairline p-6">
            <h3 className="text-sm font-semibold text-sand-800 mb-4">
              Assessment-Based Comparables ({result.comparables.length})
            </h3>
            <p className="text-xs text-sand-400 mb-4">
              Properties with similar assessed values, age, and zoning in the same neighbourhood.
              These are used by the ML model to estimate value.
            </p>
            {result.comparables.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-sand-200">
                      {["Address", "Assessed Value", "Distance", "Similarity", "Year Built", "Zoning"].map((h) => (
                        <th key={h} className="text-left py-2 px-3 text-[11px] font-medium text-sand-400 uppercase tracking-wider">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.comparables.map((comp, i) => (
                      <tr key={i} className="border-b border-sand-100 last:border-0 hover:bg-sand-50 transition">
                        <td className="py-2.5 px-3 text-sand-800">{comp.address}</td>
                        <td className="py-2.5 px-3 text-sand-700">{formatCurrencyFull(comp.assessed_value)}</td>
                        <td className="py-2.5 px-3 text-sand-500">{formatDistance(comp.distance_m)}</td>
                        <td className="py-2.5 px-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-1.5 bg-sand-100 rounded-full overflow-hidden">
                              <div className="h-full bg-teal-500 rounded-full" style={{ width: `${Math.max(5, (1 - comp.similarity_score) * 100)}%` }} />
                            </div>
                            <span className="text-[11px] text-sand-400">{((1 - comp.similarity_score) * 100).toFixed(0)}%</span>
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-sand-500">{comp.year_built || "\u2014"}</td>
                        <td className="py-2.5 px-3 text-sand-500">{comp.zoning || "\u2014"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-sand-400">No comparable properties found</p>
            )}
          </div>

          {/* Risk Flags */}
          {result.risk_flags.length > 0 && (
            <div className="card-hairline p-6">
              <h3 className="text-sm font-semibold text-sand-800 mb-4">Risk Flags</h3>
              <div className="space-y-3">
                {result.risk_flags.map((flag, i) => (
                  <div key={i} className={`flex items-start gap-3 p-3 rounded-lg border ${severityColor(flag.severity)}`}>
                    <svg className="w-4 h-4 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.27 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                    <div>
                      <div className="text-sm font-medium capitalize">{flag.category.replace(/_/g, " ")}</div>
                      <div className="text-xs mt-0.5 opacity-80">{flag.description}</div>
                    </div>
                    <span className="ml-auto text-[10px] uppercase font-semibold tracking-wider">{flag.severity}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Metadata */}
          <div className="flex items-center justify-between text-[11px] text-sand-400 px-1">
            <span>Model: {result.metadata.model_segment} &middot; v{result.metadata.model_version}</span>
            <span>{new Date(result.metadata.prediction_timestamp).toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
}


/* ================================================================
   CMA REPORT — Paragon-style Comparable Market Analysis
   ================================================================ */

function CMAReportSection({ cma }: { cma: CMAResponse | null }) {
  if (!cma) {
    return (
      <div className="card-hairline p-8 text-center">
        <p className="text-sm text-sand-400">CMA data not available for this property.</p>
      </div>
    );
  }

  if (cma.comparable_count === 0) {
    return (
      <div className="card-hairline p-8 text-center">
        <h3 className="text-sm font-semibold text-sand-800 mb-2">Comparative Market Analysis</h3>
        <p className="text-sm text-sand-400">
          No comparable sold properties found within the search area and time frame.
          Try widening the search radius or time period.
        </p>
      </div>
    );
  }

  const comps = cma.comparables;

  return (
    <div className="space-y-6">
      {/* CMA Header */}
      <div className="card-hairline p-6">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-sand-900" style={{ fontFamily: "var(--font-display)" }}>
              Comparative Market Analysis
            </h3>
            <p className="text-xs text-sand-400 mt-1">
              Recent comparable sales within 90 days &middot; {cma.comparable_count} properties found
            </p>
          </div>
          {cma.recommendation?.estimated_value && (
            <div className="text-right">
              <div className="text-xs text-sand-400 uppercase tracking-wider">Recommended Value</div>
              <div className="text-xl font-semibold text-teal-700 mt-1" style={{ fontFamily: "var(--font-display)" }}>
                {formatCurrencyFull(cma.recommendation.estimated_value)}
              </div>
              {cma.recommendation.estimated_range && (
                <div className="text-xs text-sand-400 mt-0.5">
                  {formatCurrency(cma.recommendation.estimated_range.low)} &ndash; {formatCurrency(cma.recommendation.estimated_range.high)}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Subject Property Summary */}
        <div className="bg-sand-50 rounded-lg p-4 mb-6">
          <div className="text-[11px] text-sand-400 uppercase tracking-wider mb-2">Subject Property</div>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <div>
              <div className="text-[10px] text-sand-400">Address</div>
              <div className="text-sm font-medium text-sand-800 mt-0.5">{cma.subject.address || "N/A"}</div>
            </div>
            <div>
              <div className="text-[10px] text-sand-400">Type</div>
              <div className="text-sm font-medium text-sand-800 mt-0.5 capitalize">{cma.subject.property_type || "N/A"}</div>
            </div>
            <div>
              <div className="text-[10px] text-sand-400">Beds/Baths</div>
              <div className="text-sm font-medium text-sand-800 mt-0.5">
                {cma.subject.bedrooms ?? "\u2014"} / {cma.subject.bathrooms ?? "\u2014"}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-sand-400">Floor Area</div>
              <div className="text-sm font-medium text-sand-800 mt-0.5">
                {cma.subject.floor_area ? `${Math.round(cma.subject.floor_area).toLocaleString()} sqft` : "\u2014"}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-sand-400">Year Built</div>
              <div className="text-sm font-medium text-sand-800 mt-0.5">{cma.subject.year_built ?? "\u2014"}</div>
            </div>
            <div>
              <div className="text-[10px] text-sand-400">Assessed Value</div>
              <div className="text-sm font-medium text-sand-800 mt-0.5">
                {cma.subject.assessed_value ? formatCurrencyFull(cma.subject.assessed_value) : "\u2014"}
              </div>
            </div>
          </div>
        </div>

        {/* Market Summary Stats */}
        {cma.market_stats && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
            <StatCard label="Median Sold" value={formatCurrency(cma.market_stats.median_sold_price)} />
            <StatCard label="Avg SAR" value={cma.market_stats.avg_sar ? `${(cma.market_stats.avg_sar * 100).toFixed(1)}%` : "N/A"} sub="of assessed value" />
            <StatCard label="List-to-Sold" value={cma.market_stats.avg_list_to_sold ? `${(cma.market_stats.avg_list_to_sold * 100).toFixed(1)}%` : "N/A"} />
            <StatCard label="Avg DOM" value={cma.market_stats.avg_dom ? `${cma.market_stats.avg_dom.toFixed(0)} days` : "N/A"} />
            <StatCard label="Avg Distance" value={formatDistance(cma.market_stats.avg_distance_m)} />
          </div>
        )}
      </div>

      {/* Comparable Properties Table — Paragon style */}
      <div className="card-hairline overflow-hidden">
        <div className="p-4 border-b border-sand-200 bg-sand-50/50">
          <h4 className="text-sm font-semibold text-sand-800">Comparable Sold Properties</h4>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-sand-50 border-b border-sand-200">
                {["#", "Address", "Type", "Sold Price", "List Price", "SP/LP", "Sold Date", "DOM", "Bed", "Bath", "Sqft", "Year", "Dist", "Adj. Price"].map((h) => (
                  <th key={h} className="text-left py-2.5 px-3 text-[10px] font-semibold text-sand-500 uppercase tracking-wider whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comps.map((comp, i) => (
                <CMARow key={comp.mls_number} comp={comp} index={i + 1} />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-comp Adjustment Detail */}
      <div className="card-hairline p-6">
        <h4 className="text-sm font-semibold text-sand-800 mb-4">Price Adjustments Detail</h4>
        <p className="text-xs text-sand-400 mb-4">
          Each comparable&apos;s sold price is adjusted toward the subject property to account for differences
          in age, size, and bedroom count. The adjusted prices form the CMA estimate range.
        </p>
        <div className="space-y-3">
          {comps.map((comp, i) => (
            <div key={comp.mls_number} className="border border-sand-100 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-sand-100 text-sand-600 flex items-center justify-center text-[10px] font-bold">{i + 1}</span>
                  <span className="text-sm font-medium text-sand-800">{comp.address}</span>
                  <span className="text-xs text-sand-400 font-mono">MLS# {comp.mls_number}</span>
                </div>
                <div className="text-right">
                  <span className="text-xs text-sand-400">Sold: </span>
                  <span className="text-sm font-medium text-sand-800">{formatCurrencyFull(comp.sold_price)}</span>
                  <span className="mx-2 text-sand-300">&rarr;</span>
                  <span className="text-sm font-semibold text-teal-700">{formatCurrencyFull(comp.adjusted_price)}</span>
                </div>
              </div>
              {comp.adjustments.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2">
                  {comp.adjustments.map((adj, j) => (
                    <div key={j} className="flex items-center justify-between py-1 px-3 rounded bg-sand-50 text-xs">
                      <span className="text-sand-600">{adj.name}: {adj.detail}</span>
                      <span className={`font-medium ${adj.dollar >= 0 ? "text-emerald-600" : "text-rose-500"}`}>
                        {adj.dollar >= 0 ? "+" : ""}{formatCurrency(adj.dollar)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-sand-400 mt-1">No adjustments needed</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* CMA Price Range Summary */}
      {cma.cma_range && (
        <div className="card-hairline p-6">
          <h4 className="text-sm font-semibold text-sand-800 mb-4">CMA Price Range Summary</h4>
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="text-center p-3 rounded-lg bg-sand-50">
              <div className="text-[10px] text-sand-400 uppercase">Low (25th %ile)</div>
              <div className="text-lg font-semibold text-sand-700 mt-1">{formatCurrency(cma.cma_range.low)}</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-teal-50 border border-teal-200">
              <div className="text-[10px] text-teal-600 uppercase font-medium">Median</div>
              <div className="text-lg font-semibold text-teal-700 mt-1">{formatCurrency(cma.cma_range.median)}</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-sand-50">
              <div className="text-[10px] text-sand-400 uppercase">Mean</div>
              <div className="text-lg font-semibold text-sand-700 mt-1">{formatCurrency(cma.cma_range.mean)}</div>
            </div>
            <div className="text-center p-3 rounded-lg bg-sand-50">
              <div className="text-[10px] text-sand-400 uppercase">High (75th %ile)</div>
              <div className="text-lg font-semibold text-sand-700 mt-1">{formatCurrency(cma.cma_range.high)}</div>
            </div>
          </div>

          {/* Visual range bar */}
          <div className="relative h-3 bg-sand-100 rounded-full overflow-hidden mt-4">
            <div
              className="absolute h-full bg-gradient-to-r from-teal-300 to-teal-500 rounded-full"
              style={{
                left: `${((cma.cma_range.low - cma.cma_range.low * 0.9) / (cma.cma_range.high * 1.1 - cma.cma_range.low * 0.9)) * 100}%`,
                width: `${((cma.cma_range.high - cma.cma_range.low) / (cma.cma_range.high * 1.1 - cma.cma_range.low * 0.9)) * 100}%`,
              }}
            />
          </div>
          <div className="flex justify-between mt-1.5 text-[10px] text-sand-400">
            <span>{formatCurrency(cma.cma_range.low)}</span>
            <span>{formatCurrency(cma.cma_range.high)}</span>
          </div>

          {/* Final recommendation */}
          {cma.recommendation && cma.recommendation.estimated_value && (
            <div className="mt-6 p-4 rounded-lg bg-teal-50 border border-teal-200">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-teal-700 uppercase tracking-wider">Final Recommendation</div>
                  <div className="text-sm text-teal-600 mt-1">
                    {cma.recommendation.method}
                    {cma.recommendation.note && <span className="ml-1 text-teal-500">({cma.recommendation.note})</span>}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-semibold text-teal-800" style={{ fontFamily: "var(--font-display)" }}>
                    {formatCurrencyFull(cma.recommendation.estimated_value)}
                  </div>
                  {cma.recommendation.estimated_range && (
                    <div className="text-xs text-teal-500">
                      {formatCurrency(cma.recommendation.estimated_range.low)} &ndash; {formatCurrency(cma.recommendation.estimated_range.high)}
                    </div>
                  )}
                  <div className={`inline-block mt-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider ${
                    cma.recommendation.confidence === "high" ? "bg-emerald-100 text-emerald-700" :
                    cma.recommendation.confidence === "moderate" ? "bg-amber-100 text-amber-700" :
                    "bg-rose-100 text-rose-600"
                  }`}>
                    {cma.recommendation.confidence} confidence
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CMARow({ comp, index }: { comp: CMAComparable; index: number }) {
  return (
    <tr className="border-b border-sand-100 last:border-0 hover:bg-sand-50/50 transition">
      <td className="py-2.5 px-3 text-sand-400 text-xs">{index}</td>
      <td className="py-2.5 px-3">
        <div className="text-sand-800 font-medium">{comp.address}</div>
        <div className="text-[10px] text-sand-400 font-mono">MLS# {comp.mls_number}</div>
      </td>
      <td className="py-2.5 px-3 text-sand-500 text-xs">{comp.property_type}</td>
      <td className="py-2.5 px-3 font-medium text-sand-800">{formatCurrencyFull(comp.sold_price)}</td>
      <td className="py-2.5 px-3 text-sand-500">{comp.list_price ? formatCurrencyFull(comp.list_price) : "\u2014"}</td>
      <td className="py-2.5 px-3">
        {comp.list_to_sold ? (
          <span className={`text-xs font-medium ${comp.list_to_sold >= 1 ? "text-emerald-600" : "text-rose-500"}`}>
            {(comp.list_to_sold * 100).toFixed(1)}%
          </span>
        ) : "\u2014"}
      </td>
      <td className="py-2.5 px-3 text-sand-500 text-xs whitespace-nowrap">{comp.sold_date}</td>
      <td className="py-2.5 px-3 text-sand-500">{comp.dom ?? "\u2014"}</td>
      <td className="py-2.5 px-3 text-sand-500">{comp.bedrooms ?? "\u2014"}</td>
      <td className="py-2.5 px-3 text-sand-500">{comp.bathrooms ?? "\u2014"}</td>
      <td className="py-2.5 px-3 text-sand-500">{comp.floor_area ? comp.floor_area.toLocaleString() : "\u2014"}</td>
      <td className="py-2.5 px-3 text-sand-500">{comp.year_built ?? "\u2014"}</td>
      <td className="py-2.5 px-3 text-sand-500 text-xs">{formatDistance(comp.distance_m)}</td>
      <td className="py-2.5 px-3 font-medium text-teal-700">{formatCurrencyFull(comp.adjusted_price)}</td>
    </tr>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="text-center p-3 rounded-lg bg-sand-50">
      <div className="text-[10px] text-sand-400 uppercase tracking-wider">{label}</div>
      <div className="text-sm font-semibold text-sand-800 mt-1">{value}</div>
      {sub && <div className="text-[10px] text-sand-400">{sub}</div>}
    </div>
  );
}
