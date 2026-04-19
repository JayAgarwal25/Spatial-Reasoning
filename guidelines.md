CS F425 Deep Learning – Term Project Submission Guide 
1. What every team must submit 
• A single compressed submission folder named TeamID_ProjectShortName.zip 
• A research-style final report in PDF, prepared from LaTeX source.  
• Full source code used for the final demo and experiments  
• A README with exact setup and run instructions.  
• A contribution note describing who did what within the team 
• A short demo package for the live evaluation: demo script, sample inputs, and backup video/screenshots  
2. Required folder structure 
TeamID_ProjectShortName/ 
01_admin/ 
team_info.txt 
contribution_statement.pdf 
02_report/ 
final_report.pdf 
latex_source.zip 
03_code/ 
README.md 
requirements.txt or environment.yml 
src/ 
scripts/ 
configs/ 
04_data/ 
data_description.md 
sample_inputs/ 
dataset_links.txt 
05_results/ 
main_results.csv 
ablations.csv 
figures/ 
logs/ 
06_demo/ 
demo_instructions.md 
demo_inputs/ 
backup_video.mp4   (optional but strongly recommended) 
07_claims/ 
prior_work_basis.md 
claimed_contribution.md 
Why this matters: Every team MUST stick to this structure since your submissions will be parsed using an 
automated script. Furthermore, a standardized structure it makes it easier to do uniform evaluation across 
various aspects such as understanding of prior work, implementation, experiments, demo readiness, report 
quality, and any novelty bonus. 
3. Contents of each required item 
3.1 01_admin 
• team_info.txt: team ID, member names, IDs, emails, chosen project title, faculty/TA mentor, and topic 
number/name from the project list 
• contribution_statement.pdf: 1-2 paragraphs clearly listing each member’s role in reading, coding, 
experimentation, report writing, and demo prep 
3.2 02_report 
• final_report.pdf must follow a research-paper structure: abstract, introduction, related work, 
methodology, experimental setup, results, ablations, failure cases, limitations, and references 
• The report must clearly justify any scope reduction due to compute, memory, or dataset constraints 
• The report must clearly identify the base papers studied and what the team reproduced, adapted, or 
extended 
• All citations must be verified by the team; hallucinated or copied AI-generated references can attract 
heavy penalties 
• latex_source.zip should contain the .tex sources, bibliography, and figures needed to regenerate the report 
3.3 03_code 
• README.md must include environment setup, dependency versions, hardware used, training commands, 
evaluation commands, and demo commands 
• src/ should contain the main code in a clean, readable structure 
• scripts/ should contain runnable entry points such as train, eval, infer, or demo launch scripts 
• configs/ should contain the main configuration files or hyperparameter settings 
3.4 04_data 
• data_description.md must describe the dataset used, train/val/test split, preprocessing, and any subset or 
reduced setup used 
• dataset_links.txt should include links and access notes for external datasets 
• sample_inputs/ should contain a few small, representative examples used in the demo 
3.5 05_results 
• main_results.csv should summarize the main quantitative results reported in the paper 
• ablations.csv must include at least one ablation study 
• figures/ should contain the exact plots, qualitative outputs, and tables used in the report or presentation 
• logs/ may contain training logs, evaluation logs, or notebook exports if useful for verification 
3.6 06_demo 
• demo_instructions.md must explain how to run the demo in 3–5 minutes during viva 
• The demo must show the system working end-to-end on meaningful inputs 
• backup video or screenshots are only backup materials; they do not replace the mandatory live demo 
3.7 07_claims 
• prior_work_basis.md: list the core papers read for the chosen project and briefly state how each paper 
influenced the work 
• claimed_contribution.md: state the exact project claims in plain language under these headings: What we 
reproduced; What we modified; What did not work; What we believe is our contribution 
• Be precise and honest. Clear contribution framing is better than exaggerated novelty claims. If you are 
unable to substantiate/defend the exaggerated claims, you will lose marks heavily!! 
One-page quick checklist before submission 
• Our zip file name follows the required format. 
• Our folder structure matches this guide. 
• Our report is in PDF and the LaTeX source is included. 
• Our report contains a proper related-work section and verified citations. 
• We clearly explain what we reproduced, modified, and contributed. 
• We include at least one ablation study. 
• Our README contains exact run commands. 
• Our demo can run live on meaningful inputs. 
• We include failure cases and limitations. 
• We include a contribution statement for all team members. 
