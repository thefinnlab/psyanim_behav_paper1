export default class PsyanimJsPsychTrialParameter {

    parameterType;
    entityName;
    parameterName;
    parameterValue;
    componentTypeName;

    constructor(parameterType, parameterName, parameterValue, entityName = '', componentType = '') {

        this.parameterType = parameterType;
        this.entityName = entityName;
        this.parameterName = parameterName;
        this.parameterValue = parameterValue;
        
        if (this.parameterType == PsyanimJsPsychTrialParameter.Type.COMPONENT_PARAMETER)
        {
            if (!componentType)
            {
                PsyanimDebug.error('component parameter has no component type defined!');
                return;
            }

            this.componentTypeName = componentType.name;
        }
    }

    is(trialParameter) {

        return this.parameterType == trialParameter.parameterType &&
            this.entityName == trialParameter.entityName &&
            this.parameterName == trialParameter.parameterName &&
            this.componentTypeName == trialParameter.componentTypeName;
    }
}

PsyanimJsPsychTrialParameter.Type = {
    SCENE_PARAMETER: 'scene',
    PREFAB_PARAMETER: 'prefab',
    COMPONENT_PARAMETER: 'component'
};