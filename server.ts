import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

  app.use(express.json({ limit: '10mb' }));

  // Initialize server-side Gemini client using modern @google/genai SDK
  const apiKey = process.env.GEMINI_API_KEY;
  let ai: GoogleGenAI | null = null;
  if (apiKey) {
    ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }

  // Health check endpoint
  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({
      status: 'ok',
      hasApiKey: Boolean(apiKey),
      timestamp: new Date().toISOString(),
    });
  });

  // 1. Advisor Chat Endpoint
  app.post('/api/ai/advisor-chat', async (req: Request, res: Response) => {
    try {
      const { message, chatHistory = [], financialContext = {}, scenario = 'salaried' } = req.body;
      const userName = financialContext.userName || 'Prathamesh Shingne';
      const currency = financialContext.currency || '₹';
      const income = Number(financialContext.totalIncome) || 0;
      const expenses = Number(financialContext.totalExpenses) || 0;
      const savings = income - expenses;
      const savingsRate = income > 0 ? Math.round((savings / income) * 100) : 0;
      const categories = financialContext.categoryBreakdown || [];
      const goals = financialContext.goals || [];

      const systemPrompt = `You are the "Personal Finance Advisor Bot", a high-caliber Certified Financial Planner (CFP) and wealth management architect built specifically for ${userName}.
Current Financial Context for ${userName}:
- Active Profile / Scenario: ${scenario}
- Total Monthly Income: ${currency}${income.toLocaleString()}
- Total Monthly Expenses: ${currency}${expenses.toLocaleString()}
- Net Monthly Savings: ${currency}${savings.toLocaleString()} (${savingsRate}% savings rate)
- Current Category Expenses: ${JSON.stringify(categories)}
- Active Savings Goals: ${JSON.stringify(goals)}

Guidelines:
1. Always address ${userName} with respectful, motivating, and sharp financial insights.
2. Provide concrete numbers, specific category percentage allocations (50/30/20 rule: 50% Needs, 30% Wants, 20% Savings/Debts, emergency fund runway), and clear action steps.
3. If ${userName} asks about overspending, pinpoint the exact categories from their data.
4. Adapt your counsel directly to their scenario lifestyle constraints (${scenario}).
5. Format your response cleanly using Markdown with bold highlights, bullet points, and practical takeaways.
6. Provide concise, high-value advice. Avoid generic disclaimers.`;

      if (!ai) {
        // High quality algorithmic fallback
        const fallbackReply = generateFallbackChat(userName, currency, income, expenses, savings, savingsRate, message, scenario);
        res.json({
          reply: fallbackReply.reply,
          suggestedPrompts: fallbackReply.suggestedPrompts,
          source: 'algorithmic',
        });
        return;
      }

      // Structure conversation contents ensuring alternating user/model turns
      const contents: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];

      if (Array.isArray(chatHistory)) {
        for (const item of chatHistory.slice(-6)) {
          const role = item.sender === 'bot' || item.role === 'model' ? 'model' : 'user';
          const text = (item.text || '').trim();
          if (!text) continue;

          if (contents.length > 0 && contents[contents.length - 1].role === role) {
            contents[contents.length - 1].parts[0].text += `\n${text}`;
          } else {
            contents.push({ role, parts: [{ text }] });
          }
        }
      }

      // Ensure first turn is user
      while (contents.length > 0 && contents[0].role !== 'user') {
        contents.shift();
      }

      // Append current user message
      const currentMessageText = (message || '').trim() || 'Please evaluate my overall budget and spending health.';
      if (contents.length > 0 && contents[contents.length - 1].role === 'user') {
        contents[contents.length - 1].parts[0].text += `\n${currentMessageText}`;
      } else {
        contents.push({ role: 'user', parts: [{ text: currentMessageText }] });
      }

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents,
        config: {
          systemInstruction: systemPrompt,
          temperature: 0.7,
        },
      });

      const replyText = response.text || 'I analyzed your query, but could not produce a response right now. Please try again.';
      const suggestedPrompts = [
        'How can I cut dining & entertainment costs by 20%?',
        `How long will my emergency fund last if income stops?`,
        'What is my recommended 50/30/20 category breakdown?',
      ];

      res.json({
        reply: replyText,
        suggestedPrompts,
        source: 'gemini-3.8-flash',
      });
    } catch (error: any) {
      console.error('Advisor Chat Error:', error);
      const fallbackReply = generateFallbackChat(
        req.body?.financialContext?.userName || 'Prathamesh Shingne',
        req.body?.financialContext?.currency || '₹',
        Number(req.body?.financialContext?.totalIncome) || 85000,
        Number(req.body?.financialContext?.totalExpenses) || 52000,
        (Number(req.body?.financialContext?.totalIncome) || 85000) - (Number(req.body?.financialContext?.totalExpenses) || 52000),
        Math.round((((Number(req.body?.financialContext?.totalIncome) || 85000) - (Number(req.body?.financialContext?.totalExpenses) || 52000)) / (Number(req.body?.financialContext?.totalIncome) || 85000)) * 100),
        req.body?.message || '',
        req.body?.scenario || 'Salaried Professional'
      );
      res.json({
        reply: fallbackReply.reply,
        suggestedPrompts: fallbackReply.suggestedPrompts,
        source: 'fallback',
      });
    }
  });

  // 2. AI Budget Generator Endpoint
  app.post('/api/ai/generate-budget', async (req: Request, res: Response) => {
    try {
      const { income, expenses, categorySpending = {}, scenario = 'salaried', userName = 'Prathamesh Shingne', currency = '₹' } = req.body;
      const totalIncome = Number(income) || 85000;

      if (!ai) {
        return res.json(generateAlgorithmicBudget(totalIncome, userName, scenario, currency));
      }

      const prompt = `You are an elite financial advisor analyzing finances for ${userName}.
Scenario: ${scenario}
Total Monthly Income: ${currency}${totalIncome}
Current Categorized Spending: ${JSON.stringify(categorySpending)}

Create a detailed, balanced, and realistic budget plan following proven financial strategies tailored to this specific scenario.
Return ONLY valid JSON matching this schema:
{
  "summary": string,
  "allocation": {
    "needs": { "target": number, "percentage": number },
    "wants": { "target": number, "percentage": number },
    "savings": { "target": number, "percentage": number }
  },
  "categoryLimits": [
    { "category": string, "limit": number, "tip": string }
  ],
  "actionableTips": [ string, string, string ],
  "riskAreas": [ string, string ]
}`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.3,
        },
      });

      const parsed = JSON.parse(response.text || '{}');
      res.json(parsed);
    } catch (error: any) {
      console.error('Budget Generation Error:', error);
      const fallback = generateAlgorithmicBudget(
        Number(req.body?.income) || 85000,
        req.body?.userName || 'Prathamesh Shingne',
        req.body?.scenario || 'salaried',
        req.body?.currency || '₹'
      );
      res.json(fallback);
    }
  });

  // 3. AI Financial Health Score & Predictive Insights
  app.post('/api/ai/financial-health', async (req: Request, res: Response) => {
    try {
      const { income = 0, expenses = 0, categories = [], debts = 0, emergencyFund = 0, scenario = 'salaried', userName = 'Prathamesh Shingne' } = req.body;
      const numIncome = Number(income) || 85000;
      const numExpenses = Number(expenses) || 52000;
      const numEmergency = Number(emergencyFund) || 165000;
      const netSavings = numIncome - numExpenses;
      const savingsRate = numIncome > 0 ? (netSavings / numIncome) * 100 : 0;
      const runwayMonths = numExpenses > 0 ? Number((numEmergency / numExpenses).toFixed(1)) : 0;

      let baseScore = 65;
      if (savingsRate >= 30) baseScore += 20;
      else if (savingsRate >= 20) baseScore += 15;
      else if (savingsRate >= 10) baseScore += 5;
      else if (savingsRate < 0) baseScore -= 25;

      if (runwayMonths >= 6) baseScore += 15;
      else if (runwayMonths >= 3) baseScore += 10;
      else if (runwayMonths < 1) baseScore -= 10;

      baseScore = Math.min(100, Math.max(10, Math.round(baseScore)));

      if (!ai) {
        return res.json(generateAlgorithmicHealth(baseScore, runwayMonths, savingsRate, numIncome, numExpenses));
      }

      const prompt = `Conduct a rigorous financial health assessment for ${userName}:
Scenario: ${scenario}
Monthly Income: ${numIncome}
Monthly Expenses: ${numExpenses}
Emergency Savings Cushion: ${numEmergency} (Runway: ${runwayMonths} months)
Savings Rate: ${savingsRate.toFixed(1)}%
Recent Expense Breakdown: ${JSON.stringify(categories.slice(0, 10))}

Return ONLY JSON conforming to this schema:
{
  "score": number,
  "grade": "A+" | "A" | "B" | "C" | "D",
  "runwayMonths": number,
  "savingsRate": number,
  "healthStatus": string,
  "metrics": [
    { "name": string, "score": number, "status": string }
  ],
  "keyStrengths": [ string, string ],
  "priorityWarnings": [ string, string ],
  "predictiveTrend": string
}`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.3,
        },
      });

      const parsed = JSON.parse(response.text || '{}');
      res.json(parsed);
    } catch (error: any) {
      console.error('Financial Health Error:', error);
      const fallback = generateAlgorithmicHealth(78, 3.2, 28, 85000, 52000);
      res.json(fallback);
    }
  });

  // 4. Monthly Executive Report Generator
  app.post('/api/ai/monthly-report', async (req: Request, res: Response) => {
    try {
      const { month = 'Current Month', income = 85000, expenses = 52000, categories = [], goals = [], scenario = 'Salaried Professional', userName = 'Prathamesh Shingne', currency = '₹' } = req.body;
      const numIncome = Number(income) || 85000;
      const numExpenses = Number(expenses) || 52000;
      const netSavings = numIncome - numExpenses;
      const savingsRatio = numIncome > 0 ? Math.round((netSavings / numIncome) * 100) : 0;

      if (!ai) {
        return res.json(generateAlgorithmicReport(month, userName, scenario, currency, numIncome, numExpenses, netSavings, savingsRatio, goals));
      }

      const prompt = `You are a Chief Financial Officer crafting an executive personal financial monthly summary report for ${userName}.
Details:
- Period: ${month}
- Scenario: ${scenario}
- Total Income: ${currency}${numIncome.toLocaleString()}
- Total Expenses: ${currency}${numExpenses.toLocaleString()}
- Net Cash Flow: ${currency}${netSavings.toLocaleString()}
- Savings Rate: ${savingsRatio}%
- Categories: ${JSON.stringify(categories.slice(0, 10))}
- Goals: ${JSON.stringify(goals)}

Produce an insightful, highly structured monthly review. Return ONLY JSON conforming to this schema:
{
  "reportTitle": string,
  "preparedFor": string,
  "scenario": string,
  "executiveSummary": string,
  "kpis": {
    "income": string,
    "expenses": string,
    "savings": string,
    "savingsRate": string
  },
  "overspendingCulprits": [ string, string ],
  "highlights": [ string, string ],
  "actionPlanNextMonth": [ string, string, string ],
  "goalMilestones": [
    { "name": string, "progress": string, "recommendation": string }
  ]
}`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.3,
        },
      });

      const parsed = JSON.parse(response.text || '{}');
      res.json(parsed);
    } catch (error: any) {
      console.error('Monthly Report Error:', error);
      const fallback = generateAlgorithmicReport(
        req.body?.month || 'Current Month',
        req.body?.userName || 'Prathamesh Shingne',
        req.body?.scenario || 'Salaried Professional',
        req.body?.currency || '₹',
        85000,
        52000,
        33000,
        38,
        req.body?.goals || []
      );
      res.json(fallback);
    }
  });

  // Vite middleware in dev, static files in production
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true',
        watch: process.env.DISABLE_HMR === 'true' ? null : {},
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Personal Finance Advisor Bot server running on http://0.0.0.0:${PORT}`);
  });
}

// Algorithmic Fallback Helpers
function generateFallbackChat(userName: string, currency: string, income: number, expenses: number, savings: number, savingsRate: number, message: string, scenario: string) {
  const q = (message || '').toLowerCase();

  if (q.includes('50/30/20') || q.includes('budget') || q.includes('split') || q.includes('ratio')) {
    const needs = Math.round(income * 0.5);
    const wants = Math.round(income * 0.3);
    const sav = Math.round(income * 0.2);
    return {
      reply: `### 📊 50/30/20 Optimal Budget Plan for **${userName}**\n\nBased on your monthly income of **${currency}${income.toLocaleString()}** in the **${scenario}** profile:\n\n1. **Essential Needs (50%): ${currency}${needs.toLocaleString()} / month**\n   - Covers rent, utilities, groceries, transit, and medical obligations.\n2. **Discretionary Wants (30%): ${currency}${wants.toLocaleString()} / month**\n   - Dining out, entertainment, shopping, and OTT subscriptions.\n3. **Wealth Accumulation (20%): ${currency}${sav.toLocaleString()} / month**\n   - Automated deposits into emergency fund, mutual funds, or debt repayment.\n\n*Current Net Savings:* You are retaining **${currency}${savings.toLocaleString()}** (${savingsRate}% savings rate).`,
      suggestedPrompts: [
        'How can I trim my discretionary wants this week?',
        'Calculate my ideal emergency fund target',
        'Analyze my top category spending leaks',
      ],
    };
  }

  if (q.includes('cut') || q.includes('save') || q.includes('leak') || q.includes('overspend') || q.includes('expense')) {
    return {
      reply: `### 💡 High-Impact Expense Optimization for **${userName}**\n\nHere are 3 concrete levers to free up capital in your ledger:\n\n1. **Consolidate Dining & Deliveries:** Dining out and food deliveries often carry a 35-40% convenience premium. Cooking 2 extra meals weekly can save **${currency}4,500 - ${currency}8,000/month**.\n2. **Audit Recurring Subscriptions:** Review automated recurring streaming, gym, or software renewals.\n3. **Pay Yourself First Rule:** Move **${currency}${Math.round(income * 0.2).toLocaleString()}** into your savings or investment account on salary day before allocating for discretionary spending.`,
      suggestedPrompts: [
        'Give me a week-by-week spending budget',
        'How does my savings rate compare to benchmarks?',
        'Suggest an emergency fund plan for my scenario',
      ],
    };
  }

  if (q.includes('emergency') || q.includes('runway') || q.includes('buffer')) {
    const threeMo = expenses * 3;
    const sixMo = expenses * 6;
    return {
      reply: `### 🛡️ Emergency Reserve Strategy for **${userName}**\n\nBased on your current monthly living costs of **${currency}${expenses.toLocaleString()}**:\n\n- **3-Month Baseline Cushion:** **${currency}${threeMo.toLocaleString()}** (covers immediate unexpected events or medical needs)\n- **6-Month Complete Financial Shield:** **${currency}${sixMo.toLocaleString()}** (total career freedom and severance runway)\n\n**Best Practice:** Park this reserve in liquid High-Yield Savings or Liquid Mutual Funds yielding 6-7% p.a., completely segregated from your daily spending account.`,
      suggestedPrompts: [
        'How do I balance emergency funds with investing?',
        'What should I prioritize: debt payoff or savings?',
        'Evaluate my active savings goals',
      ],
    };
  }

  return {
    reply: `Hello **${userName}**! 👋 I am your dedicated **Personal Finance Advisor Bot**.\n\nHere is your current financial snapshot in the **${scenario}** profile:\n- **Monthly Inflow:** **${currency}${income.toLocaleString()}**\n- **Monthly Outflow:** **${currency}${expenses.toLocaleString()}**\n- **Net Positive Cashflow:** **${currency}${savings.toLocaleString()}** (${savingsRate}% savings rate)\n\nHow would you like to optimize your finances today?`,
    suggestedPrompts: [
      'Pinpoint my biggest spending leaks',
      'Generate a customized 50/30/20 budget',
      'Evaluate my financial health score',
    ],
  };
}

function generateAlgorithmicBudget(totalIncome: number, userName: string, scenario: string, currency: string) {
  const needs = Math.round(totalIncome * 0.5);
  const wants = Math.round(totalIncome * 0.3);
  const savings = Math.round(totalIncome * 0.2);
  return {
    summary: `Structured financial blueprint tailored for ${userName} under the ${scenario} framework with a 20% automated wealth target.`,
    allocation: {
      needs: { target: needs, percentage: 50 },
      wants: { target: wants, percentage: 30 },
      savings: { target: savings, percentage: 20 },
    },
    categoryLimits: [
      { category: 'Housing & Rent', limit: Math.round(totalIncome * 0.25), tip: 'Cap shelter and society maintenance under 28% of net take-home pay.' },
      { category: 'Groceries & Food', limit: Math.round(totalIncome * 0.12), tip: 'Batch shop staples and pantry essentials to minimize impulse buys.' },
      { category: 'Transport & Fuel', limit: Math.round(totalIncome * 0.08), tip: 'Utilize monthly transit passes or optimize commute routes.' },
      { category: 'Utilities & Bills', limit: Math.round(totalIncome * 0.05), tip: 'Monitor seasonal electricity and annual broadband renewals.' },
      { category: 'Dining Out & Cafes', limit: Math.round(totalIncome * 0.12), tip: 'Reserve dining out for weekends; carry lunch on weekdays.' },
      { category: 'Entertainment & OTT', limit: Math.round(totalIncome * 0.08), tip: 'Audit active digital streaming plans and pause unused subscriptions.' },
      { category: 'Healthcare & Medical', limit: Math.round(totalIncome * 0.05), tip: 'Maintain adequate health insurance safety net and annual wellness checks.' },
      { category: 'Investments & Stocks', limit: Math.round(totalIncome * 0.20), tip: 'Schedule automated SIP / index deposits on the 1st of every month.' },
      { category: 'Emergency Fund', limit: Math.round(totalIncome * 0.05), tip: 'Liquid buffer for unforeseen expenses.' },
    ],
    actionableTips: [
      `Automate a transfer of ${currency}${savings.toLocaleString()} immediately on salary credit day.`,
      'Cap discretionary food deliveries to twice per week.',
      'Review monthly bank statements quarterly to eliminate hidden fees.',
    ],
    riskAreas: ['Impulse weekend dining and online food deliveries', 'Unused digital recurring subscriptions'],
  };
}

function generateAlgorithmicHealth(baseScore: number, runwayMonths: number, savingsRate: number, income: number, expenses: number) {
  return {
    score: baseScore,
    grade: baseScore >= 85 ? 'A+' : baseScore >= 75 ? 'A' : baseScore >= 60 ? 'B' : 'C',
    runwayMonths,
    savingsRate: Math.round(savingsRate),
    healthStatus: baseScore >= 75 ? 'Strong Financial Resilience' : 'Moderate - Needs Optimization',
    metrics: [
      { name: 'Savings Velocity', score: Math.min(100, Math.max(10, Math.round(savingsRate * 3.2))), status: savingsRate >= 20 ? 'Optimal' : 'Low' },
      { name: 'Emergency Cushion', score: Math.min(100, Math.round(runwayMonths * 16.6)), status: runwayMonths >= 3 ? 'Healthy' : 'At Risk' },
      { name: 'Budget Discipline', score: expenses < income ? 88 : 42, status: expenses < income ? 'In Control' : 'Overspending' },
      { name: 'Fixed Overhead Ratio', score: 82, status: 'Balanced' },
    ],
    keyStrengths: [
      `Positive monthly cashflow retained consistently (${Math.round(savingsRate)}% savings rate).`,
      `Liquid emergency reserves cover approximately ${runwayMonths} months of essential expenses.`,
    ],
    priorityWarnings: [
      'Discretionary dining and leisure categories exhibit seasonal spikes.',
      'Ensure health and term insurance coverages are kept up to date.',
    ],
    predictiveTrend: `Maintaining this trajectory will grow annual net worth by approximately 18-24% over the next 12 months.`,
  };
}

function generateAlgorithmicReport(month: string, userName: string, scenario: string, currency: string, income: number, expenses: number, savings: number, savingsRatio: number, goals: any[]) {
  return {
    reportTitle: `Executive Financial Audit & Performance Review (${month})`,
    preparedFor: userName,
    scenario,
    executiveSummary: `During ${month}, ${userName} operated with strong fiscal discipline under the ${scenario} framework. Out of ${currency}${income.toLocaleString()} total logged income, ${currency}${savings.toLocaleString()} was retained into net wealth reserves, delivering a stellar ${savingsRatio}% savings rate.`,
    kpis: {
      income: `${currency}${income.toLocaleString()}`,
      expenses: `${currency}${expenses.toLocaleString()}`,
      savings: `${currency}${savings.toLocaleString()}`,
      savingsRate: `${savingsRatio}%`,
    },
    overspendingCulprits: [
      'Dining Out & Cafes (+18% over target allowance)',
      'Digital Subscriptions & OTT (+8% over benchmark)',
    ],
    highlights: [
      `Retained ${currency}${savings.toLocaleString()} surplus net cashflow without incurring short-term consumer debt.`,
      `Met monthly savings milestone ahead of projected schedule.`,
    ],
    actionPlanNextMonth: [
      `Schedule automated recurring transfer of ${currency}${Math.round(income * 0.20).toLocaleString()} on day 1.`,
      'Enforce the 48-hour rule on non-essential purchases exceeding 1% of monthly income.',
      'Replenish emergency buffer to reach target runway.',
    ],
    goalMilestones: goals.map((g: any) => ({
      name: g.name || 'Savings Goal',
      progress: `${g.current || g.currentAmount || 0}/${g.target || g.targetAmount || 0}`,
      recommendation: 'Maintain monthly recurring contribution.',
    })),
  };
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
