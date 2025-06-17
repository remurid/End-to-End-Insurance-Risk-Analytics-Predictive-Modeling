def interpret_result(test_result: dict, alpha=0.05) -> str:
    """Interpret the result of a statistical test."""
    if test_result['p_value'] < alpha:
        return f"Reject the null hypothesis (p = {test_result['p_value']:.4f})."
    else:
        return f"Fail to reject the null hypothesis (p = {test_result['p_value']:.4f})."

def business_recommendation(hypothesis: str, result: dict, group_a: str, group_b: str) -> str:
    """Provide a business recommendation based on the test result."""
    if result['p_value'] < 0.05:
        return f"We reject the null hypothesis for {hypothesis}. There is a significant difference between {group_a} and {group_b}. Consider adjusting segmentation or pricing accordingly."
    else:
        return f"No significant difference found for {hypothesis} between {group_a} and {group_b}. No change recommended."
