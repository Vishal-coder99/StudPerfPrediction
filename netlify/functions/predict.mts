import type { Config } from "@netlify/functions";

// Parameters exported from the trained scikit-learn models in /models
// (StandardScaler + LinearRegression + LogisticRegression), so predictions
// match the Python Flask server exactly.
const MEAN = [5.62655, 72.19149999999999, 64.8405, 7.00775, 6.0275];
const SCALE = [2.6121877511963034, 15.786462642086732, 14.93952257436629, 1.1875878651704048, 3.751565506558562];
const REG_COEF = [7.158789677880364, 4.032789287404524, 5.233240628583832, 0.7543321930706457, 4.6994565541762805];
const REG_INTERCEPT = 65.380875;
const CLF_COEF = [1.7169744457279894, 1.5511287481152387, 1.6321044227489392, 0.16304093544971424, 1.5961730030273111];
const CLF_INTERCEPT = 8.003982249717062;

const num = (v: unknown, fallback: number) => {
  const n = Number(v ?? fallback);
  if (!Number.isFinite(n)) throw new Error("Invalid numeric input");
  return n;
};

export default async (req: Request) => {
  if (req.method !== "POST") {
    return Response.json({ error: "Method not allowed" }, { status: 405 });
  }
  try {
    const data = await req.json();
    const study = num(data.study_hours, 5.0);
    const attendance = num(data.attendance, 75.0);
    const prev = num(data.previous_marks, 60.0);
    const sleep = num(data.sleep_hours, 7.0);
    const tests = Math.trunc(num(data.practice_tests, 3));

    const scaled = [study, attendance, prev, sleep, tests].map((x, i) => (x - MEAN[i]) / SCALE[i]);
    const dot = (w: number[]) => w.reduce((s, wi, i) => s + wi * scaled[i], 0);

    const rawMarks = dot(REG_COEF) + REG_INTERCEPT;
    const predictedMarks = Math.round(Math.max(0, Math.min(100, rawMarks)) * 10) / 10;

    const z = dot(CLF_COEF) + CLF_INTERCEPT;
    const p = 1 / (1 + Math.exp(-z));
    const passProbability = Math.round(p * 1000) / 10;
    const status = p >= 0.5 ? "PASS" : "FAIL";

    let advice: string;
    if (attendance < 75) {
      advice = "Attendance is critically low (<75%). Attend upcoming lectures to prevent exam debarment.";
    } else if (study < 3.0) {
      advice = "Daily study hours are low. Increasing study by 1.5 hours daily will boost score by ~4 marks.";
    } else if (predictedMarks < 40) {
      advice = "High academic risk detected! Urgent remedial classes and mock exam revision required.";
    } else {
      advice = "Student demonstrates strong consistency. Recommend continuing regular revision sessions.";
    }

    return Response.json({
      status,
      predicted_marks: predictedMarks,
      pass_probability: passProbability,
      advice,
    });
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : String(e) }, { status: 400 });
  }
};

export const config: Config = { path: "/predict" };
