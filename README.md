A Python calculus-powered graphing app 
It lets you input a function, plot the function and its derivative, and compute a definite integral over a chosen interval.

## Features

- Function input using Python syntax (example: `x**2 + 3*x + 5`)
- Numerical derivative (central difference)
- Numerical integration using `scipy.integrate.quad`
- Matplotlib visualization of:
  - original function
  - derivative
  - shaded integral area
- Tkinter GUI controls for ranges and integration bounds
- Save graph as image (`.png` or `.jpg`)

## Setup

1. Create virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Run

```powershell
python main.py
```

## Suggested Demo Script

1. Enter `x**2 + 3*x + 5`
2. Set `x min = -10`, `x max = 10`, `points = 500`
3. Set integral bounds `a = 0`, `b = 5`
4. Click **Plot Function**
5. Explain:
   - blue line is `f(x)`
   - red line is numerical derivative `f'(x)`
   - green area is integral region
6. Click **Save Graph** to export output

