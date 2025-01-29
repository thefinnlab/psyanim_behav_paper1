import {

    PsyanimConstants,

    PsyanimPlayfightFSM,
    PsyanimPlayfightSeparationFSM,
    PsyanimPlayfightHFSM,

    PsyanimFSMStateRecorder

} from 'psyanim2';

// import PsyanimConstants from '../node_modules/psyanim2/src/core/PsyanimConstants.js';
// import PsyanimPlayfightFSM from '../node_modules/psyanim2/src/core/components/ai/playfight/PsyanimPlayfightFSM.js'
// import PsyanimPlayfightSeparationFSM from '../node_modules/psyanim2/src/core/components/ai/playfight/PsyanimPlayfightSeparationFSM.js'
// import PsyanimPlayfightHFSM from '../node_modules/psyanim2/src/core/components/ai/playfight/PsyanimPlayfightHFSM.js'

// In this scene with 2 agents playing/fighting, I'm going to call one the agent1 
// and one the agent2 - they're both playing/fighting


/**
 *  Parameters below here are the same for both agents, and defined above so we don't have to update
 *  each agent individually every time we want to change one parameter to observe its effect.
 */


// arrive behavior
const maxChargeSpeed = 3;
const maxChargeAcceleration = 0.5;

// charge state
const maxChargeDuration = 2000;

// wander state
const breakDurationAverage = 2000;
const breakDurationVariance = 200;

const minTargetDistanceForCharge = 200;
const maxTargetDistanceForCharge = 500; //400; //450 backup;

const wanderPanicDistance = 800; // 250 for the versions shared on Jan30; 450 may work/ be better too. changed to 450 on Feb9

const wanderFleeRate = 0;

const wanderSensorRadiusPadding = 150; //75

// charge delay state
const averageChargeDelay = 200; // 100;
const chargeDelayVariance = 100; // what I had: 400, but smaller the better

const minWanderDuration = averageChargeDelay + chargeDelayVariance + 100; // 1000; see notes above

// wander behavior
const maxWanderSpeed = 1.5;
const maxWanderAcceleration = 0.1;

// separation state
const maxSeparationSpeed = 1.5;
const maxSeparationAcceleration = 0.1;
const maxSeparationDuration = 1000; // 750 backup //100;

export default {
    key: 'playfightScene',
    wrapScreenBoundary: false,
    entities: [{
            name: 'agent1',
            initialPosition: { x: 250, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 12,
                color: 0x000000 //0x00ff00
            },
            components: [{
                    type: PsyanimPlayfightFSM,
                    params: {

                        // wander state
                        breakDurationAverage: breakDurationAverage,
                        breakDurationVariance: breakDurationVariance,
                        minWanderDuration: minWanderDuration,
                        minTargetDistanceForCharge: minTargetDistanceForCharge,
                        maxTargetDistanceForCharge: maxTargetDistanceForCharge,

                        wanderFleeOrChargeWhenAttacked: true, //true,
                        wanderPanicDistance: wanderPanicDistance,
                        wanderFleeRate: wanderFleeRate,

                        wanderSensorRadiusPadding: wanderSensorRadiusPadding,

                        // flee state
                        maxFleeDuration: 100,

                        // charge state
                        maxChargeDuration: maxChargeDuration,

                        // charge delay state (after detecting a charge, "preparing for a response")
                        // introduced to reduce mirroring
                        averageChargeDelay: averageChargeDelay,
                        chargeDelayVariance: chargeDelayVariance,

                        // arrive behavior
                        maxChargeSpeed: maxChargeSpeed,
                        maxChargeAcceleration: maxChargeAcceleration,

                        innerDecelerationRadius: 12,
                        outerDecelerationRadius: 30,

                        // wander behavior
                        maxWanderSpeed: maxWanderSpeed,
                        maxWanderAcceleration: maxWanderAcceleration, //0.2,
                        wanderRadius: 50,
                        wanderOffset: 250,
                        maxWanderAngleChangePerFrame: 20,

                        // flee behavior
                        maxFleeSpeed: 12,
                        maxFleeAcceleration: 0.5,
                        fleePanicDistance: 200,

                        target: {
                            entityName: 'agent2',
                        },
                    }
                },
                {
                    type: PsyanimPlayfightSeparationFSM,
                    params: {
                        target: {
                            entityName: 'agent2'
                        },

                        maxSeparationSpeed: maxSeparationSpeed, //9,
                        maxSeparationAcceleration: maxSeparationAcceleration, //0.3,
                    }
                },
                {
                    type: PsyanimPlayfightHFSM,
                    params: {

                        maxSeparationDuration: maxSeparationDuration,

                        playfightFSM: {
                            entityName: 'agent1',
                            componentType: PsyanimPlayfightFSM
                        },
                        separationFSM: {
                            entityName: 'agent1',
                            componentType: PsyanimPlayfightSeparationFSM
                        },
                        //debug: false //true
                        debugLogging: false,
                        debugGraphics: false
                    }
                },
                {
                    type: PsyanimFSMStateRecorder,
                    params: {

                        stateMachine: {
                            entityName: 'agent1',
                            componentType: PsyanimPlayfightHFSM
                        },

                        saveResumeEventSnapshot: true,
                        savePauseEventSnapshot: true,
                        saveStopEventSnapshot: true,
                        saveEnterEventSnapshot: true,
                        saveExitEventSnapshot: true,
                    }
                }
            ]
        },
        {
            name: 'agent2',
            initialPosition: { x: 550, y: 300 },
            shapeParams: {
                shapeType: PsyanimConstants.SHAPE_TYPE.CIRCLE,
                radius: 12,
                color: 0xcccccc
            },
            components: [{
                    type: PsyanimPlayfightFSM,
                    params: {

                        // wander state
                        breakDurationAverage: breakDurationAverage,
                        breakDurationVariance: breakDurationVariance,
                        minWanderDuration: minWanderDuration,
                        minTargetDistanceForCharge: minTargetDistanceForCharge,
                        maxTargetDistanceForCharge: maxTargetDistanceForCharge,

                        wanderFleeOrChargeWhenAttacked: true, //false: not flee OR charge,
                        wanderPanicDistance: wanderPanicDistance,
                        wanderFleeRate: wanderFleeRate,

                        wanderSensorRadiusPadding: wanderSensorRadiusPadding,

                        // flee state
                        maxFleeDuration: 100,

                        // charge state
                        maxChargeDuration: maxChargeDuration,

                        // charge delay state
                        averageChargeDelay: averageChargeDelay,
                        chargeDelayVariance: chargeDelayVariance,

                        // arrive behavior
                        maxChargeSpeed: maxChargeSpeed,
                        maxChargeAcceleration: maxChargeAcceleration,

                        innerDecelerationRadius: 12,
                        outerDecelerationRadius: 30,

                        // wander behavior
                        maxWanderSpeed: maxWanderSpeed,
                        maxWanderAcceleration: maxWanderAcceleration,
                        wanderRadius: 50,
                        wanderOffset: 250,
                        maxWanderAngleChangePerFrame: 20,

                        // flee behavior
                        maxFleeSpeed: 12,
                        maxFleeAcceleration: 0.5,
                        fleePanicDistance: 200,

                        target: {
                            entityName: 'agent1'
                        },
                    }
                },
                {
                    type: PsyanimPlayfightSeparationFSM,
                    params: {

                        target: {
                            entityName: 'agent1'
                        },

                        maxSeparationSpeed: maxSeparationSpeed,
                        maxSeparationAcceleration: maxSeparationAcceleration,
                    }
                },
                {
                    type: PsyanimPlayfightHFSM,
                    params: {

                        maxSeparationDuration: maxSeparationDuration,

                        playfightFSM: {
                            entityName: 'agent2',
                            componentType: PsyanimPlayfightFSM
                        },
                        separationFSM: {
                            entityName: 'agent2',
                            componentType: PsyanimPlayfightSeparationFSM
                        },
                        debugLogging: false,
                        debugGraphics: false
                    }
                },
                {
                    type: PsyanimFSMStateRecorder,
                    params: {

                        stateMachine: {
                            entityName: 'agent2',
                            componentType: PsyanimPlayfightHFSM
                        },

                        saveResumeEventSnapshot: true,
                        savePauseEventSnapshot: true,
                        saveStopEventSnapshot: true,
                        saveEnterEventSnapshot: true,
                        saveExitEventSnapshot: true,
                    }
                }

            ]
        }
    ]
};