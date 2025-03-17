from smolagents.memory import SystemPromptStep, TaskStep, ActionStep, PlanningStep
from smolagents.monitoring import LogLevel
import wandb

def replay(agent):
        logger = agent.logger
        logger.console.log("#" * 25 + "Replaying the agent's steps:" + "#" * 25)
        logger.console.log(agent.memory.system_prompt.to_messages(summary_mode=False))
        log_level=LogLevel.ERROR
        for i,step in enumerate(agent.memory.steps):
            if isinstance(step, SystemPromptStep):
                logger.log_rule(f"System Prompt Step", level=log_level)

            elif isinstance(step, TaskStep):
                logger.log_rule(f"Task Step", level=log_level)

            elif isinstance(step, ActionStep):
                logger.log_rule(f"Action Step {step.step_number}", level=log_level)

            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=log_level)

            logger.console.log(step.to_messages(summary_mode=False))
        
        total_tokens = agent.monitor.get_total_token_counts()
        wandb.log({
            "total_tokens": total_tokens['input'] + total_tokens['output'],
            "input_tokens": total_tokens['input'],
            "output_tokens": total_tokens['output']
        })