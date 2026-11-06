// Serverless port of server.py's /predict endpoint.
// Weights are exported from the trained scikit-learn models in models/*.joblib
// (StandardScaler + LinearRegression + LogisticRegression).
const MODEL = {
  mean: [5.62655, 72.1915, 64.8405, 7.00775, 6.0275],
  scale: [2.6121877511963034, 15.786462642086732, 14.93952257436629, 1.1875878651704048, 3.751565506558562],
  regCoef: [7.158789677880364, 4.032789287404524, 5.233240628583832, 0.7543321930706457, 4.6994565541762805],
  regIntercept: 65.380875,
  clfCoef: [1.7169744457279894, 1.5511287481152387, 1.6321044227489392, 0.16304093544971424, 1.5961730030273111],
  clfIntercept: 8.003982249717062,
};

const num = (v, d) => (Number.isFinite(Number(v)) && v !== null && v !== "" ? Number(v) : d);

export default async (req) => {
  if (req.method !== "POST") {
    return Response.json({ error: "Use POST" }, { status: 405 });
  }
  try {
    const data = await req.json();
    const study = num(data.study_hours, 5.0);
    const attendance = num(data.attendance, 75.0);
    const prev = num(data.previous_marks, 60.0);
    const sleep = num(data.sleep_hours, 7.0);
    const tests = Math.trunc(num(data.practice_tests, 3));

    const x = [study, attendance, prev, sleep, tests].map((v, i) => (v - MODEL.mean[i]) / MODEL.scale[i]);
    const dot = (w, b) => x.reduce((s, v, i) => s + v * w[i], b);

    let predMarks = Math.max(0, Math.min(100, dot(MODEL.regCoef, MODEL.regIntercept)));
    predMarks = Math.round(predMarks * 10) / 10;

    const p = 1 / (1 + Math.exp(-dot(MODEL.clfCoef, MODEL.clfIntercept)));
    const passProb = Math.round(p * 1000) / 10;
    const status = p >= 0.5 ? "PASS" : "FAIL";

    let advice;
    if (attendance < 75) {
      advice = "Attendance is critically low (<75%). Attend upcoming lectures to prevent exam debarment.";
    } else if (study < 3.0) {
      advice = "Daily study hours are low. Increasing study by 1.5 hours daily will boost score by ~4 marks.";
    } else if (predMarks < 40) {
      advice = "High academic risk detected! Urgent remedial classes and mock exam revision required.";
    } else {
      advice = "Student demonstrates strong consistency. Recommend continuing regular revision sessions.";
    }

    return Response.json({ status, predicted_marks: predMarks, pass_probability: passProb, advice });
  } catch (e) {
    return Response.json({ error: String(e?.message ?? e) }, { status: 400 });
  }
};

export const config = { path: "/predict" };
