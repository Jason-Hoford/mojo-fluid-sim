# Tutorial: Installing Mojo (and Python + pygame) on Windows using WSL and pixi

This tutorial shows you how to:

- Install **WSL** (Ubuntu on Windows)
- Install **pixi**
- Install **Mojo** inside a project environment
- Add **Python + pygame** to the same environment
- Use **VS Code** to edit and run your Mojo / Python projects

---

## 1. Check if WSL is installed

Open **Windows PowerShell** (normal, not admin) and run:

```powershell
wsl -l -v
```

- If you see a list of installed Linux distributions (for example, `Ubuntu`), WSL is already installed.  
- If you get an error, follow the steps below to install WSL.

---

## 2. Install WSL (if needed)

1. Open **Windows PowerShell as Administrator**  
   - Press <Win>, type **PowerShell**, right-click **Windows PowerShell**, choose **Run as administrator**.

2. In the admin PowerShell window, run:

   ```powershell
   wsl --install
   ```

3. Wait for the installation to finish, then **restart your computer** when asked.

4. After the restart, Windows will open a window to finish setting up your Linux distro (usually **Ubuntu**).  
   - Create a **username** and **password** for your Linux account.

---

## 3. Start WSL

Open **Windows PowerShell** (normal) and run:

```powershell
wsl
```

You should now see a prompt like:

```bash
yourname@PC:/mnt/c/Users/yourname$
```

This means you are inside Linux (WSL).

---

## 4. Install pixi

In the WSL terminal, run:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

After installation finishes, check that pixi is available:

```bash
eval "$(pixi completion --shell bash)"
```

(If this command runs without an error, pixi is installed correctly.)

---

## 5. Create a Mojo project with pixi

In WSL, go to your Windows home folder (optional but recommended):

```bash
cd /mnt/c/Users/yourname
```

Now create a new project:

```bash
pixi init hello-world \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd hello-world
```

This will:

- Create a folder called `hello-world`
- Create a `pixi.toml` file
- Move you into the `hello-world` directory

Next, add Mojo to the project:

```bash
pixi add mojo
```

Then enter the project environment:

```bash
pixi shell
```

You should see something like:

```bash
(hello-world) yourname@PC:/mnt/c/Users/yourname/hello-world$
```

---

## 6. Verify Mojo is installed

Inside the pixi shell, run:

```bash
mojo --version
```

You should see a version number, for example:

```text
Mojo 0.26.x.x ...
```

If you see a version, Mojo is installed correctly in this pixi environment.

---

## 7. Open the project in VS Code

Still inside the pixi shell and in the `hello-world` folder, run:

```bash
code .
```

This will:

- Open **VS Code** on your Windows machine
- Attach it to the current WSL folder

> Make sure you have VS Code installed and the **Remote - WSL** extension enabled.

In VS Code, you can now:

- Create a file called `hello.mojo`
- Edit Mojo and Python files in this project

---

## 8. Running Mojo code

Create a file `hello.mojo` in your project with this content:

```mojo
fn main():
    print("Hello, World!")
```

Then, in the **pixi shell** (inside `hello-world`):

```bash
mojo hello.mojo
```

You should see:

```text
Hello, World!
```

---

## 9. Add Python and pygame to the same environment

If you want to use Python and pygame in this project (for example, for your 2D fluid simulator), add them via pixi:

```bash
pixi add python pygame
```

Now, inside the pixi shell, you can run:

```bash
python -c "import pygame; print(pygame.__version__)"
```

If this prints a version number, **pygame** is installed correctly.

You can now create and run Python files, for example:

```bash
python fluid_sim_2d.py
```

---

## 10. Summary of common commands

From Windows PowerShell:

```powershell
wsl                             # enter Linux
```

Inside WSL:

```bash
cd /mnt/c/Users/yourname/hello-world
pixi shell                      # enter the project environment

mojo --version                  # check Mojo
mojo hello.mojo                 # run a Mojo file

python fluid_sim_2d.py          # run a Python file (with pygame)
code .                          # open folder in VS Code
```

To add more Python libraries later:

```bash
pixi add <library-name>
```

For example:

```bash
pixi add numpy
```

---

## 11. Official Mojo installation docs

For more detailed or advanced installation options (including other platforms), see the official documentation:

- https://docs.modular.com/mojo/manual/install/
